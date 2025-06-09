#
# Copyright IBM Corp 2024
#

import copy
import functools
import operator
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import sendnn
import torch
import torch.fx
from sendnn import opcodes
from torch._dynamo.backends.common import aot_autograd
from torch._guards import TracingContext
from torch.distributed.distributed_c10d import get_rank, get_world_size
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.utils._mode_utils import no_dispatch

from torch_sendnn import _logging
from torch_sendnn.decomposition import sendnn_decompositions
from torch_sendnn.sendnnops import prevent_op_decomp
from torch_sendnn.utils import SpyreGraphCache, convert

log = _logging.get_logger(__name__)
event_counter = _logging.get_event_counter()

aten = torch.ops.aten
# TODO:
# - memory layout
# - tensor layout
# LOG
# format
# getitem
def convert_from_sendnn_tensor(t):
    return convert.convert_from_sendnn_tensor(t)


def convert_from_sendnn_data_type(dt):
    return convert.convert_from_sendnn_datatype(dt)


def convert_to_sendnn_tensor(inp, datatype=sendnn.sen_datatype_enum.dt_undef, layout=sendnn.TensorLayout.NCHW):
    return convert.convert_to_sendnn_tensor(inp, datatype, layout)


def convert_tensor_to_sendnn_tensor(inp):
    return convert.convert_tensor_to_sendnn_tensor(inp)


def convert_shape(shape):
    return convert.convert_shape(shape, IS_DYNAMIC)


def convert_data_type(dtype):
    return convert.convert_datatype(dtype)

# layout is required for inputs and outputs of conv, pooling, biasAdd
# TODO: require layout for outputs only? so that layout only needs to be set for these ops
def convert_layout(shape: list):
    rank = len(shape)
    if rank == 5:
        return sendnn.TensorLayout.NCDHW
    else:
        return sendnn.TensorLayout.NCHW

# [Note: Usage of SymInt.node._hint for dynamic shapes]
# Context: Dynamic shapes are represented in pytorch as a torch.SymInt object, 
#   which holds a formula (e.g. "s0 + 1"), and a hint (e.g. "s0 = 64").
# Objective: For our use case, we want to grab both formula and hint and get a number back. 
#   Thankfully, SymInts can be casted to int through int(SymInt object), 
#   which does exactly what we want.
# Problem: on torch >= 2.4.0, if we do int(symint), it works and no guards are created. 
#   On 2.3.1, our currently officially supported version, doing the same creates a guard 
#   that essentially turns this graph tracing into a static graph as far as dynamo is concerned. 
# Debugging: To find about this, I had to look through the 2.3.1 source code, and find there is a 
#   torch.fx.experimental._config.print_specializations = True call that tells you where 
#   these kind of guard creations happen, as otherwise we were completely oblivious about it.
# Solution: Once I found the issue (the calls to int()), the alternative has been to look at what is
#   being done internally at the int() call, which so happens to be a call to a cached value
#   for the expression called symint.node._hint that has the information we need. This is standard
#   practice for these kinds of symbolic engines, and so we should not expect the API to change.
# Future: We might want to create a nicer API that either uses int(SymInt) if supported properly,
#   or use the hint instead if still on older versions.


class MyShapeProp:
    def __init__(self, mod: GraphModule, shape_env: ShapeEnv|None):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.shape_env = shape_env

    def propagate(self, *args):
        args_iter = iter(args)
        env = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            result = None
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'output':
                # Nothing needs to be propagated for output. If its shape did need to be propagated,
                # it would have been recorded via a node which feeds into output
                continue

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype
                node.is_contiguous = result.is_contiguous()
            elif isinstance(result, tuple) or isinstance(result, list):
                node.shape = []
                node.dtype = []
                for res in result:
                    if res is not None:
                        node.shape.append(res.shape)
                        node.dtype.append(res.dtype)

            elif isinstance(result, torch.SymInt):
                # See [Note: Usage of SymInt.node._hint for dynamic shapes]
                if IS_DYNAMIC:
                    node.value = result
                else:
                    if self.shape_env is not None:
                        with self.shape_env.suppress_guards():
                            node.value = result.node._hint
            elif isinstance(result, int):
                node.value = result
            else:
                raise TypeError(f"Unsupported ShapeProp type {type(result)}")

            env[node.name] = result

model_inputs = []

def is_bwd_graph(graph) -> bool:
    is_bwd = False
    for node in graph.nodes:
        if node.op == "placeholder" and "tangents" in node.name:
            is_bwd = True
            break
    return is_bwd

class FxToSenDnn:
    """
    This class handles the translation between FX graphs coming from AOTAutograd and SenDNN graphs

    Inputs:
    gm: The FX graph from AOTAutograd that will be translated
    fake_tensor_inputs: The list of inputs to the graph generated by AOTAutograd

    """

    DEVICE_MISMATCH_ERROR = "On AIU, the device must match the original tensor device"

    def __init__(
        self,
        gm: GraphModule,
        fake_tensor_inputs: list[torch.Tensor],
        static_input_indices: list,
        is_grad_enabled: bool,
        is_optim: bool,
    ):
        log.info("Original FX Graph: %s", gm.print_readable(False))
        self.gb = sendnn.GraphBuilder()
        self.sendnn_nodes = {}
        self.gm = gm
        self.fake_tensor_inputs = fake_tensor_inputs
        self.args_iter = iter(fake_tensor_inputs)
        self.args_idx = 0
        self.pi_idx = 0
        self.mi_idx = 0
        # TODO: discern MI and PI
        self.static_input_indices = static_input_indices
        self.is_grad_enabled = is_grad_enabled
        self.is_bwd = is_bwd_graph(gm.graph)
        self.is_optim = is_optim
        # pprint.pprint(self.shapes)
        self.input_idx = 0
        # TODO: mean and var of BN are model inputs
        self.bn_inputs = []
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and str(node.target)
                == "aten._native_batch_norm_legit_no_training.default"
            ):
                self.bn_inputs.append(node.args[3].name)
                self.bn_inputs.append(node.args[4].name)

    def find_inputs(self, node: torch.fx.Node):
        # TODO(mschaal): Ensure stride, memory_format in tensor_meta all make sense and are compatible
        inputs = []
        inp_idx = 0
        for x in node.args:
            if type(x) == torch.fx.node.Node:
                inputs.append(self.sendnn_nodes[x.name])
            elif type(x) == torch.fx.immutable_collections.immutable_list:
                # TODO: create separate list?
                for n in x:
                    if type(n) == torch.fx.node.Node:
                        if IS_DYNAMIC and ("sym_size_int" in n.name):
                            continue
                        inputs.append(self.sendnn_nodes[n.name])
            inp_idx = inp_idx + 1
            self.input_idx = self.input_idx + 1

        return inputs

    def convert_placeholder(self, node):
        inp = next(self.args_iter)

        # convert SymInt nodes as well, might be needed for dynamic shape
        # TODO: add SymInt to Convert function

        if isinstance(inp, torch.SymInt):
            self.args_idx += 1
            if IS_DYNAMIC:
                node_name = str(node.value)
                if (node_name in self.sendnn_nodes):
                    return self.sendnn_nodes[node_name]
                inp_shape = (1,)
                shape = sendnn.TensorShape(inp_shape)
                dt_float32 = sendnn.sen_datatype_enum.int64
                layout = convert_layout(inp_shape)
                ti = sendnn.TensorInfo(dt_float32, shape, layout)
                node.name = node_name
                return self.gb.PrimaryInput(node_name, ti)
            else:
                # can this be removed for static shape?
                return self.convert_sym_size(node, [])

        self.args_idx += 1
        dt_float32 = convert_data_type(inp.dtype)
        inp_shape = inp.shape
        if not inp_shape:
            inp_shape = [1]
        shape = convert_shape(inp_shape)
        layout = convert_layout(inp_shape)
        ti = sendnn.TensorInfo(dt_float32, shape, layout)
        # TODO: discern MI and PI based on current AIU support (subject to change)
        # Inference: arg#_1 (params, buffers, inputs) 
        #            params+buffers -> MIs, inputs -> PIs (according to self.static_input_indices)
        # Training: fwd: primals_# (params, buffers, inputs) 
        #                inputs -> PI (according to self.static_input_indices)
        #                (params+buffers).requires_grad=False -> MIs, requires_grad=True -> PIs
        #           bwd: primals_#, intermediate, tangents (according to fwd?)
        #           opt: arg#_1 (params, states, grads) -> PIs
        global model_inputs  
        def conv_mi(node):
            node_name = "mi_" + str(self.mi_idx)
            self.mi_idx += 1
            node.name = node_name
            return self.gb.ModelInput(node_name, ti)

        def conv_pi(node):
            node_name = "pi_" + str(self.pi_idx)
            self.pi_idx += 1
            node.name = node_name
            return self.gb.PrimaryInput(node_name, ti)

        if self.is_bwd:
            if node.name in model_inputs:
                return conv_mi(node)
            else:
                return conv_pi(node)
        elif self.is_optim:
            # opt currently fallback to cpu
            return conv_pi(node)
        elif not self.is_grad_enabled:
            idx = self.args_idx - 1
            if idx in self.static_input_indices:
                return conv_mi(node)
            else:
                return conv_pi(node)
        else:
            idx = self.args_idx - 1
            if idx not in self.static_input_indices:
                return conv_pi(node)
            else:
                if inp.requires_grad:
                    return conv_pi(node)
                else:
                    model_inputs.append("mi_" + str(self.mi_idx))
                    return conv_mi(node)

    def convert_getattr(self, node):
        with no_dispatch():
            attr = getattr(self.gm, node.target)  # 0-dim tensor
            sendnn_tensor = convert_to_sendnn_tensor(attr)
            return self.gb.ConstInput(node.name, sendnn_tensor)

    @staticmethod
    def convert_singular_function(gb_fn):
        def conv_fn(fx_to_sendnn, node, inputs):
            dt = convert_data_type(node)
            shape = convert_shape(node)
            layout = convert_layout(node.shape)
            ti = sendnn.TensorInfo(dt, shape, layout)
            return gb_fn(fx_to_sendnn.gb, node.name, ti, inputs[0])

        return conv_fn

    def convert_get_item(self, node, inputs):
        indexed_node = sendnn.NodeOrIndexedNode(node.args[1], inputs[0])

        return indexed_node

    @staticmethod
    def convert_binary_function(gb_fn):
        def conv_fn(fx_to_sendnn, node, inputs):
            for i, arg in enumerate(node.args):

                if type(arg) in [float, int, bool]:
                    shape = convert_shape([1])
                    layout = sendnn.TensorLayout.NCHW
                    if type(arg) is float:
                        dt = sendnn.sen_datatype_enum.float32
                        ti = sendnn.TensorInfo(dt, shape, layout)
                        consttensor = sendnn.ConvertToConstTensorFloat32(ti, [arg])
                    elif type(arg) is int:
                        dt = sendnn.sen_datatype_enum.int64
                        ti = sendnn.TensorInfo(dt, shape, layout)
                        consttensor = sendnn.ConvertToConstTensorInt64(ti, [arg])
                    else:
                        raise TypeError(f"Unsupported datatype {type(arg)} for convert_toConstTensor")
                    constinput = fx_to_sendnn.gb.ConstInput(node.name + "_const_" + str(i), consttensor)
                    inputs.insert(i, constinput)
            dt = convert_data_type(node)
            shape = convert_shape(node)
            layout = convert_layout(node.shape)
            ti = sendnn.TensorInfo(dt, shape, layout)
            return gb_fn(fx_to_sendnn.gb, node.name, ti, inputs[0], inputs[1])

        return conv_fn

    def convert_all_gather(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        rank_set = node.args[2]
        rank = get_rank()
        world_size = get_world_size()
        return self.gb.AllGather(node.name, ti, inputs[0], rank, rank_set, world_size)

    # Native functional collective format.
    # Introduced in:
    # https://github.com/pytorch/pytorch/pull/120370
    # (INPUT, GROUP_SIZE, GROUP_NAME)
    # Example: (clone, 2, 'default')
    def convert_all_gather_native(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        group_name = node.args[2]
        if group_name == "default":
            pg = torch.distributed.group.WORLD
        else:
            pg = torch._C._distributed_c10d._resolve_process_group(group_name)
        rank_set = torch.distributed.distributed_c10d.get_process_group_ranks(pg)
        rank = get_rank()
        world_size = get_world_size()
        return self.gb.AllGather(node.name, ti, inputs[0], rank, rank_set, world_size)

    def convert_all_reduce(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        reduce_op = node.args[1]
        rank_set = node.args[3]
        rank = get_rank()
        world_size = get_world_size()
        if reduce_op == "sum":
            return self.gb.AllReduceSum(node.name, ti, inputs[0], rank, rank_set, world_size)
        else:
            raise TypeError(f"Unsupported AllReduce op {reduce_op}")

    # Native functional collective format.
    # Introduced in
    # https://github.com/pytorch/pytorch/pull/120370
    # (INPUT, OP, GROUP_NAME)
    # Example: (view_23, 'sum', 'default')
    def convert_all_reduce_native(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        reduce_op = node.args[1]
        group_name = node.args[2]
        if group_name == "default":
            pg = torch.distributed.group.WORLD
        else:
            pg = torch._C._distributed_c10d._resolve_process_group(group_name)
        rank_set = torch.distributed.distributed_c10d.get_process_group_ranks(pg)
        rank = get_rank()
        world_size = get_world_size()
        if reduce_op == "sum":
            return self.gb.AllReduceSum(node.name, ti, inputs[0], rank, rank_set, world_size)
        else:
            raise TypeError(f"Unsupported AllReduce op {reduce_op}")

    def convert_reduce_scatter_native(self, node, inputs):
        # Reduces the tensor data across all machines, then scatter the results
        # to corresponding ranks.
        # inputs: self: torch.Tensor,
        #         reduceOp: str,
        #         group_size: int,
        #         group_name: str
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        reduce_op = node.args[1]
        group = node.args[3]
        if group == "default":
            pg = torch.distributed.group.WORLD
        else:
            pg = torch._C._distributed_c10d._resolve_process_group(group)
        rank_set = torch.distributed.distributed_c10d.get_process_group_ranks(pg)
        rank = get_rank()
        world_size = get_world_size()
        if reduce_op == "sum":
            return self.gb.ReduceScatterSum(node.name, ti, inputs[0], rank, rank_set, world_size)
        elif reduce_op == "avg":
            return self.gb.ReduceScatterAvg(node.name, ti, inputs[0], rank, rank_set, world_size)
        else:
            raise TypeError(f"Unsupported ReduceScatter op {reduce_op}")

    def convert_wait_tensor(self, node, inputs):
        if isinstance(inputs[0], sendnn.NodeOrIndexedNode):
            # For nested wait calls, just return the prior generated node
            return inputs[0]
        indexed_node = sendnn.NodeOrIndexedNode(inputs[0])
        return indexed_node

    @staticmethod
    def convert_pooling(gb_fn):
        # https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/Pooling.cpp#L110
        #     const Tensor& self,
        # IntArrayRef kernel_size,
        # IntArrayRef stride,
        # IntArrayRef padding,
        # IntArrayRef dilation,
        # bool ceil_mode
        def pooling_conv_fn(fx_to_sendnn, node, inputs):
            dt = convert_data_type(node.dtype[0])
            shape = convert_shape(node.shape[0])
            layout = convert_layout(node.shape[0])
            ti = sendnn.TensorInfo(dt, shape, layout)
            kernel_size = convert_shape(node.args[1])
            stride = convert_shape(node.args[2])
            padding = sendnn.PaddingInfo()
            ph = pw = 0
            if len(node.args) >= 4:
                ph, pw = node.args[3]
            padding.begin = sendnn.TensorShape([0, 0, ph, pw])
            padding.end = sendnn.TensorShape([0, 0, ph, pw])
            dilations = convert_shape(torch.Size([1, 1, 1, 1]))

            return gb_fn(fx_to_sendnn.gb, node.name, ti, inputs[0], kernel_size, stride, padding, dilations)

        return pooling_conv_fn

    @staticmethod
    def convert_avg_pooling(gb_fn):
        # https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/Pooling.cpp#L110
        #     const Tensor& self,
        # IntArrayRef kernel_size,
        # IntArrayRef stride,
        # IntArrayRef padding,
        # IntArrayRef dilation,
        # bool ceil_mode
        def pooling_conv_fn(fx_to_sendnn, node, inputs):
            dt = convert_data_type(node.dtype)
            shape = convert_shape(node.shape)
            layout = convert_layout(node.shape)
            ti = sendnn.TensorInfo(dt, shape, layout)
            kernel_size = convert_shape(node.args[1])
            stride = convert_shape(node.args[2])
            padding = sendnn.PaddingInfo()
            ph = pw = 0
            if len(node.args) >= 4:
                ph, pw = node.args[3]
            padding.begin = sendnn.TensorShape([0, 0, ph, pw])
            padding.end = sendnn.TensorShape([0, 0, ph, pw])
            return gb_fn(fx_to_sendnn.gb, node.name, ti, inputs[0], kernel_size, stride, padding, True, 0)

        return pooling_conv_fn

    @staticmethod
    def convert_matmul(gb_fn):
        def matmul_conv_fn(fx_to_sendnn, node, inputs):
            dt = convert_data_type(node)
            shape = convert_shape(node)
            layout = convert_layout(node.shape)
            ti = sendnn.TensorInfo(dt, shape, layout)
            return gb_fn(fx_to_sendnn.gb, node.name, ti, *inputs, False, False)

        return matmul_conv_fn

    def convert_batch_matmul_w4a16(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.BatchMatMul_W4A16(node.name, ti, *inputs)

    def convert_batch_matmul_w8a8(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = sendnn.TensorLayout.NCHW
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.BatchMatMul_W8A8(node.name, ti, *inputs, node.args[-3], node.args[-2], node.args[-1])

    def convert_unknown(self, node, inputs):
        log.debug("->  Unknown: ", node)
        if isinstance(node.dtype, list):
            dt = [convert_data_type(t) for t in node.dtype]
            shape = [convert_shape(s) for s in node.shape]
            layout = [convert_layout(s) for s in node.shape]
            ti = [sendnn.TensorInfo(t, s, layout) for t, s in zip(dt, shape)]
        else:
            dt = convert_data_type(node.dtype)
            shape = convert_shape(node.shape)
            layout = convert_layout(node.shape)
            ti = [sendnn.TensorInfo(dt, shape, layout)]

        #====================================================
        # Modification for catching unsupported ops
        #====================================================

        def add_prefix_to_string(original_string):
            prefix = "DEEPVIEW "
            return '\n'.join(prefix + line for line in original_string.split('\n'))

        unsup_op = os.environ.get('UNSUP_OP', "0")
        unsup_op_debug = os.environ.get('UNSUP_OP_DEBUG', "0")
        if unsup_op == '1':
            error = ""
            if unsup_op_debug == '1':
                error = f"DEEPVIEW==================================== Stack Trace ====================================\n{add_prefix_to_string(node.stack_trace)}"
            print(f"DEEPVIEW Caught error for \033[1m{node}\033[0m: Operation not supported.\nDEEPVIEW Data type: {dt}, Shape: {shape}\n{error}")
        #====================================================

        return self.gb.UnknownNode(node.name, ti, inputs)

    def convert_addmm(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        alpha = 1.0
        beta = 1.0
        return self.gb.AddMm(node.name, ti, *inputs, alpha, beta)

    def convert_arange(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        start_value = 0
        end_value = 1
        step_value = 1
        if len(node.args) == 1:
            end_value = node.args[0]
        elif len(node.args) == 2:
            start_value = node.args[0]
            end_value = node.args[1]
        elif len(node.args) == 3:
            start_value = node.args[0]
            end_value = node.args[1]
            step_value = node.args[2]
        start_dt = sendnn.sen_datatype_enum.int64
        start_shape = convert_shape([1])
        start_ti = sendnn.TensorInfo(start_dt, start_shape, sendnn.TensorLayout.UNDEF)
        start_tensor = sendnn.ConvertToConstTensorInt64(start_ti, [start_value])
        start = self.gb.ConstInput(node.name + "_start", start_tensor)
        end_tensor = sendnn.ConvertToConstTensorInt64(start_ti, [end_value])
        end = self.gb.ConstInput(node.name + "_end", end_tensor)
        step_tensor = sendnn.ConvertToConstTensorInt64(start_ti, [step_value])
        step = self.gb.ConstInput(node.name + "_step", step_tensor)
        return self.gb.Arange(node.name, ti, start, end, step)

    def convert_batch_norm(self, node, inputs):
        output_shapes = list()
        num_outputs = len(node.users)
        for i in range(num_outputs):
            dt = convert_data_type(node.dtype[i])
            shape = convert_shape(node.shape[i])
            layout = convert_layout(node.shape[i])
            ti = sendnn.TensorInfo(dt, shape, layout)
            output_shapes.append(ti)
        return self.gb.FusedBatchNorm(node.name, output_shapes, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                      node.args[6])

    def convert_cat(self, node, inputs):
        dt = convert_data_type(node)
        dim = node.args[1] if len(node.args) > 1 else 0
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.Concat(node.name, ti, inputs, dim)

    def convert_clamp(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        min_value = np.finfo(np.float32).min
        max_value = np.finfo(np.float32).max
        if len(node.args) >= 2:
            if node.args[1] is not None:
                min_value = node.args[1]
            if len(node.args) == 3:
                max_value = node.args[2]
        return self.gb.Clip(node.name, ti, inputs[0], min_value, max_value)

    def convert_convolution(self, node, inputs):
        # https://github.com/zdevito/ATen/blob/master/aten/src/ATen/native/Convolution.cpp#L516
        #
        # at::Tensor convolution(
        #     const Tensor& input, const Tensor& weight, const Tensor& bias,
        # IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
        # bool transposed, IntArrayRef output_padding, int64_t groups) {
        dtype = convert_data_type(node)
        shape = convert_shape(node)
        bias = node.args[2]
        stride = convert_shape(node.args[3])
        pad = node.args[4]
        dilation = convert_shape(node.args[5])
        transpose = node.args[6]
        groups = node.args[8]
        padding = sendnn.PaddingInfo()
        padding.begin = sendnn.TensorShape([0, 0] + pad)
        padding.end = sendnn.TensorShape([0, 0] + pad)
        ret_list = []
        conv_name = node.name if bias is None else node.name + "_conv"
        ti = sendnn.TensorInfo(dtype, shape, convert_layout(node.shape))
        if transpose:
            output = self.gb.ConvolutionTranspose(conv_name, ti, inputs[0], inputs[1], stride, padding, dilation,
                                                  groups)
        else:
            output = self.gb.Convolution(conv_name, ti, inputs[0], inputs[1], stride, padding, dilation, groups)
        ret_list.append([conv_name, output])
        if bias is not None:
            newbias = self.gb.BiasAdd(node.name, ti, output, inputs[2])
            ret_list.append([node.name, newbias])
        return ret_list

    def convert_cumsum(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        const_value = node.args[1]
        const_dt = sendnn.sen_datatype_enum.int64
        const_shape = convert_shape([1])
        const_layout = sendnn.TensorLayout.NCHW
        const_ti = sendnn.TensorInfo(const_dt, const_shape, const_layout)
        const_tensor = sendnn.ConvertToConstTensorInt64(const_ti, [const_value])
        const_input = self.gb.ConstInput(node.name + "_const_1", const_tensor)
        inputs.append(const_input)
        return self.gb.CumSum(node.name, ti, inputs[0], inputs[1], False, False)

    def convert_dropout(self, node, inputs):
        output_shapes = list()
        for i in range(2):
            dt = convert_data_type(node.dtype[i])
            shape = convert_shape(node.shape[i])
            layout = convert_layout(node.shape[i])
            ti = sendnn.TensorInfo(dt, shape, layout)
            output_shapes.append(ti)
        return self.gb.Dropout(node.name, output_shapes, inputs[0], node.args[1])

    def convert_dropout_backward(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.DropoutBackward(node.name, ti, inputs[0], inputs[1], node.args[2])

    def convert_embedding(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.Gather(node.name, ti, inputs[0], inputs[1], 0, 0)

    def convert_embedding_backward(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.EmbeddingBackward(node.name, ti, inputs[0], inputs[1], 
                                         node.args[2], node.args[3], node.args[4])

    def convert_expand_dims(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.ExpandDims(node.name, ti, inputs[0], node.args[1])

    def convert_full(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        
        dims_arg = node.args[0]
        value_arg = node.args[1]
        if node.target == aten.new_full.default:
            dims_arg = node.args[1]
            value_arg = node.args[2]

        dims_dt = sendnn.sen_datatype_enum.int64
        dims_shape = convert_shape(dims_arg)
        dims_ti = sendnn.TensorInfo(dims_dt, dims_shape, sendnn.TensorLayout.UNDEF)
        dims_tensor = sendnn.ConvertToConstTensorInt64(dims_ti, dims_arg)
        dims = self.gb.ConstInput(node.name + "_dims", dims_tensor)

        value_tensor = convert_to_sendnn_tensor(value_arg)
        value = self.gb.ConstInput(node.name + "_value", value_tensor)
        return self.gb.Fill(node.name, ti, dims, value)

    def convert_greater_equal(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        if type(node.args[1]) is float:
            inp_dt = sendnn.sen_datatype_enum.float32
            inp_shape = convert_shape([1])
            inp_layout = sendnn.TensorLayout.NCHW
            inp_ti = sendnn.TensorInfo(inp_dt, inp_shape, inp_layout)
            inp_tensor = sendnn.ConvertToConstTensorFloat32(inp_ti, node.args[1])
            inp = self.gb.ConstInput(node.name + "_input_1", inp_tensor)
            inputs.insert(1, inp)
        return self.gb.GreaterEqual(node.name, ti, inputs[0], inputs[1])

    def convert_gelu(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        approximate = False
        if len(node.args) == 2 and node.args[1] == "tanh":
            approximate = True
        return self.gb.Gelu(node.name, ti, inputs[0], approximate)

    def convert_gelu_backward(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        approximate = False
        if len(node.args) == 2 and node.args[1] == "tanh":
            approximate = True
        return self.gb.GeluBackward(node.name, ti, inputs[0], inputs[1], approximate)

    def convert_index(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        # TODO: is it correct mapping?
        if len(inputs) != 2:
            log.warning("Index without exactly 2 index tensors is not supported")
            return self.convert_unknown(node, inputs)
        
        return self.gb.Gather(node.name, ti, inputs[0], inputs[1], 0, 0)

    def convert_layer_norm(self, node, inputs):
        output_shapes = list()
        num_outputs = len(node.users)
        for i in range(num_outputs):
            dt = convert_data_type(node.dtype[i])
            shape = convert_shape(node.shape[i])
            layout = convert_layout(node.shape[i])
            ti = sendnn.TensorInfo(dt, shape, layout)
            output_shapes.append(ti)
        axis = []
        for i in range(len(node.args[1])):
            axis_i = -1 - i
            axis.append(axis_i)
        axis_shape = convert_shape(axis)
        return self.gb.LayerNorm(node.name, output_shapes, inputs[0], inputs[1], inputs[2], axis_shape, node.args[4])

    def convert_layer_norm_backward(self, node, inputs):
        # inputs: grad_output: Tensor
        #         input: Tensor
        #         normalized_shape: []
        #         mean: Tensor
        #         rstd: Tensor
        #         weight: Optional[Tensor]
        #         bias: Optional[Tensor]
        #         output_mask: bool[3]
        # outputs: [Tensor, Tensor, Tensor]
        output_shapes = list()
        num_outputs = len(node.users)
        for i in range(num_outputs):
            dt = convert_data_type(node.dtype[i])
            shape = convert_shape(node.shape[i])
            layout = convert_layout(node.shape[i])
            ti = sendnn.TensorInfo(dt, shape, layout)
            output_shapes.append(ti)
        axis = []
        for i in range(len(node.args[2])):
            axis_i = -1 - i
            axis.append(axis_i)
        axis_shape = convert_shape(axis)
        output_mask = convert_shape(node.args[7])
        return self.gb.LayerNormBackward(node.name, output_shapes, inputs[0], inputs[1], inputs[2], 
                                         inputs[3], inputs[4], inputs[5], axis_shape, output_mask)

    def convert_log_softmax(self, node, inputs):
        # https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
        # inputs: input: Tensor
        #         dim: int64_t
        #         half_to_float: bool
        # outputs: Tensor
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.LogSoftmax(node.name, ti, inputs[0], node.args[1])

    def convert_log_softmax_backward(self, node, inputs):
        # inputs: grad: Tensor
        #         output: Tensor
        #         dim: int64_t
        #         input_dtype: ScalarType
        # outputs: Tensor
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.LogSoftmaxBackward(node.name, ti, inputs[0], inputs[1], node.args[2])

    def convert_new_empty(self, node, inputs):
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        
        orig_tensor_arg = node.args[0]

        if "dtype" in node.kwargs:
            dtype_arg = node.kwargs["dtype"]
        else:
            dtype_arg = orig_tensor_arg.dtype

        if "device" in node.kwargs:
            if node.kwargs["device"] != orig_tensor_arg.device:
                raise ValueError(self.DEVICE_MISMATCH_ERROR)

        ti = sendnn.TensorInfo(convert_data_type(dtype_arg), shape, layout)
        return self.gb.Empty(node.name, ti)

    def convert_new_zeros(self, node, inputs):
        shape = convert_shape(node)
        layout = convert_layout(node.shape)

        orig_tensor_arg = node.args[0]
        size_arg = node.args[1]

        if "dtype" in node.kwargs:
            dtype_arg = node.kwargs["dtype"]
        else:
            dtype_arg = orig_tensor_arg.dtype

        if "device" in node.kwargs:
            if node.kwargs["device"] != orig_tensor_arg.device:
                raise ValueError(self.DEVICE_MISMATCH_ERROR)

        ti = sendnn.TensorInfo(convert_data_type(dtype_arg), shape, layout)
        dims_shape = convert_shape(size_arg)
        return self.gb.Zeros(node.name, ti, dims_shape)
    
    def convert_new_ones(self, node, inputs):
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        
        orig_tensor_arg = node.args[0]
        size_arg = node.args[1]

        if "dtype" in node.kwargs:
            dtype_arg = node.kwargs["dtype"]
        else:
            dtype_arg = orig_tensor_arg.dtype

        if "device" in node.kwargs:
            if node.kwargs["device"] != orig_tensor_arg.device:
                raise ValueError(self.DEVICE_MISMATCH_ERROR)

        ti = sendnn.TensorInfo(convert_data_type(dtype_arg), shape, layout)
        dims_shape = convert_shape(size_arg)
        return self.gb.Ones(node.name, ti, dims_shape)

    def convert_nll_loss(self, node, inputs):
        # https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html
        # inputs: self: Tensor
        #         target: Tensor
        #         weight: Optional[Tensor] (None by default)
        #         reduction: int (0 - Reduction::None, 1 - Reduction::Mean, 2 - Reduction::Sum), 
        #         ignore_index: int (-100)
        # outputs: result, total_weight: tuple[Tensor, Tensor]
        out_ti = []
        for idx in range(2):
            dt = convert_data_type(node.dtype[idx])
            shape = convert_shape(node.shape[idx])
            layout = convert_layout(node.shape[idx])
            ti = sendnn.TensorInfo(dt, shape, layout)
            out_ti.append(ti)

        if node.args[2] is not None:
            log.warning(f"Expected node.args[2] to be None, instead got {node.args[2]}")
            return self.convert_unknown(node, inputs)

        # The values for node.args[3] are the ones corresponding to the reduction enum
        # described in https://github.com/pytorch/pytorch/blob/main/torch/nn/_reduction.py#L8
        if node.args[3] == 0:  # reduction is "none"
            return self.gb.NllLoss(node.name, out_ti, inputs[0], inputs[1], node.args[4])
        elif node.args[3] == 1:  # reduction is "mean" (default)
            return self.gb.NllLossMean(node.name, out_ti, inputs[0], inputs[1], node.args[4])
        elif node.args[3] == 2:  # reduction is "sum"
            return self.gb.NllLossSum(node.name, out_ti, inputs[0], inputs[1], node.args[4])

    def convert_nll_loss_backward(self, node, inputs):
        # inputs: grad_output: Tensor
        #         self: Tensor
        #         target: Tensor
        #         weight: Optional[Tensor]
        #         reduction: int (0 - Reduction::None, 1 - Reduction::Mean, 2 - Reduction::Sum)
        #         ignore_index: int
        #         total_weight: Tensor
        # outputs: grad_input: Tensor
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)

        if node.args[3] is not None:
            log.warning(f"Expected node.args[3] to be None, instead got {node.args[3]}")
            return self.convert_unknown(node, inputs)

        # The values for node.args[4] are the ones corresponding to the reduction enum
        # described in https://github.com/pytorch/pytorch/blob/main/torch/nn/_reduction.py#L8
        if node.args[4] == 0:  # reduction is "none"
            return self.gb.NllLossBackward(node.name, ti, inputs[0], inputs[1], inputs[2], inputs[3], node.args[5])
        elif node.args[4] == 1:  # reduction is "mean" (default)
            return self.gb.NllLossMeanBackward(node.name, ti, inputs[0], inputs[1], inputs[2], inputs[3], node.args[5])
        elif node.args[4] == 2:  # reduction is "sum"
            return self.gb.NllLossSumBackward(node.name, ti, inputs[0], inputs[1], inputs[2], inputs[3], node.args[5])

    def convert_norm(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)

        if node.args[1] != 2:
            log.warning(f"Expected node.args[1] to be 2 as this maps to ReduceEuclideanNorm (only 2-norm is allowed). Instead got {node.args[1]}")
            return self.convert_unknown(node, inputs)

        dim = convert_shape(node.args[2])
        keepdim = node.args[3]
        return self.gb.ReduceEuclideanNorm(node.name, ti, inputs[0], dim, keepdim)

    def convert_ones(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        shape = convert_shape(node.args[0])
        return self.gb.Ones(node.name, ti, shape)

    def convert_pow(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        for i in range(len(node.args)):
            node_arg = node.args[i]
            if type(node_arg) != torch.fx.Node:
                inp_tensor = convert_to_sendnn_tensor(node_arg)
                inp = self.gb.ConstInput(node.name + "_input_" + str(i), inp_tensor)
                inputs.insert(i, inp)
        return self.gb.Pow(node.name, ti, inputs[0], inputs[1])

    def convert_reduce_mean(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        axis = convert_shape(node.args[1])
        keep_dims = node.args[2] if len(node.args) > 2 else False
        return self.gb.ReduceMean(node.name, ti, inputs[0], axis, keep_dims)

    def convert_repeat(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        repeats = convert_shape(node.args[1])
        return self.gb.Tile(node.name, ti, inputs[0], repeats)

    def convert_reshape(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        out_shape = convert_shape(node.args[1])
        return self.gb.Reshape(node.name, ti, inputs[0], out_shape)

    def convert_resize_nearest(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        height, width = node.args[1]
        return self.gb.ResizeNearestNeighbor(node.name, ti, inputs[0], int(height), int(width), False)
    
    def convert_resize_nearest_sendnn(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        height, width = node.args[2]
        return self.gb.ResizeNearestNeighbor(node.name, ti, inputs[0], int(height), int(width), False)

    def convert_scalar_tensor(self, node, inputs):
        with no_dispatch():
            scalar_tensor = torch.tensor(node.args[0], dtype=node.dtype)
            const_tensor = convert_tensor_to_sendnn_tensor(scalar_tensor)
            return self.gb.ConstInput(node.name, const_tensor)

    def convert_slice(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        inp_shape = node.args[0].shape
        inp_rank = len(inp_shape)
        begin = np.zeros(inp_rank, dtype=np.int64)
        end = np.array(inp_shape, dtype=np.int64)
        strides = np.ones(inp_rank, dtype=np.int64)
        # Only the first arg is required
        # FIXME(thalexan): Can axis be negative to reflect from the right?
        axis = node.args[1] if len(node.args) > 1 else 0
        if len(node.args) > 2:
            begin[axis] = node.args[2]
        if len(node.args) > 3:
            end[axis] = node.args[3]
        if len(node.args) > 4:
            strides[axis] = node.args[4]
        const_dt = sendnn.sen_datatype_enum.int64
        const_shape = convert_shape([inp_rank])
        const_layout = sendnn.TensorLayout.NCHW
        const_ti = sendnn.TensorInfo(const_dt, const_shape, const_layout)
        begin_tensor = sendnn.ConvertToConstTensorInt64(const_ti, begin)
        end_tensor = sendnn.ConvertToConstTensorInt64(const_ti, end)
        strides_tensor = sendnn.ConvertToConstTensorInt64(const_ti, strides)
        begin_input = self.gb.ConstInput(node.name + "_const_1", begin_tensor)
        end_input = self.gb.ConstInput(node.name + "_const_2", end_tensor)
        strides_input = self.gb.ConstInput(node.name + "_const_3", strides_tensor)
        return self.gb.StridedSlice(node.name, ti, inputs[0], begin_input, end_input, strides_input, 0, 0, 0, 0, 0)

    def convert_slice_backward(self, node, inputs):
        # inputs: grad_output: Tensor
        #         input_sizes: [int]
        #         dim: int
        #         start: int
        #         end: int
        #         step: int
        # outputs: grad_input: Tensor
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        inp_shape = node.args[1]
        inp_rank = len(inp_shape)
        begin = np.zeros(inp_rank, dtype=np.int64)
        end = np.array(inp_shape, dtype=np.int64)
        strides = np.ones(inp_rank, dtype=np.int64)
        axis = node.args[2]
        begin[axis] = node.args[3]
        end[axis] = node.args[4]
        strides[axis] = node.args[5]
        const_dt = sendnn.sen_datatype_enum.int64
        const_shape = convert_shape([inp_rank])
        const_layout = sendnn.TensorLayout.NCHW
        const_ti = sendnn.TensorInfo(const_dt, const_shape, const_layout)
        begin_tensor = sendnn.ConvertToConstTensorInt64(const_ti, begin)
        end_tensor = sendnn.ConvertToConstTensorInt64(const_ti, end)
        strides_tensor = sendnn.ConvertToConstTensorInt64(const_ti, strides)
        begin_input = self.gb.ConstInput(node.name + "_const_1", begin_tensor)
        end_input = self.gb.ConstInput(node.name + "_const_2", end_tensor)
        strides_input = self.gb.ConstInput(node.name + "_const_3", strides_tensor)
        return self.gb.StridedSliceBackward(node.name, ti, inputs[0], begin_input, end_input, strides_input)
    
    def convert_silu_backward(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.SiluBackward(node.name, ti, inputs[0], inputs[1])

    def convert_select_backward(self, node, inputs):
        # inputs: grad_output: Tensor
        #         input_sizes: [int]
        #         dim: int
        #         index: int
        # outputs: grad_input: Tensor
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        dim = node.args[2]
        index = node.args[3]
        return self.gb.SelectBackward(node.name, ti, inputs[0], dim, index)

    def convert_softmax(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        # args[2]: half_to_float?
        return self.gb.Softmax(node.name, ti, inputs[0], node.args[1])

    def convert_softmax_backward(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        # args[3]: at::ScalarType input_dtype?
        return self.gb.SoftmaxBackward(node.name, ti, inputs[0], inputs[1], node.args[2])

    def convert_split(self, node, inputs):
        out_ti = []

        for idx in range(len(node.shape)):
            dt = convert_data_type(node.dtype[idx])
            shape = convert_shape(node.shape[idx])
            layout = convert_layout(node.shape[idx])
            ti = sendnn.TensorInfo(dt, shape, layout)
            out_ti.append(ti)
        splits = convert_shape(node.args[1])
        dim = node.args[2] if len(node.args) > 2 else 0
        return self.gb.Split(node.name, out_ti, inputs[0], splits, dim)

    def convert_squeeze(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        axis = convert_shape(node.args[1])
        return self.gb.Squeeze(node.name, ti, inputs[0], axis)

    def convert_stack(self, node, inputs):
        dt = convert_data_type(node)
        dim = node.args[1] if len(node.args) > 1 else 0
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.Stack(node.name, ti, inputs, dim)

    def convert_sum(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        axes = convert_shape(node.args[1])
        keep_dims = True
        if len(node.shape) < len(node.args[0].shape):
            keep_dims = False
        return self.gb.ReduceSum(node.name, ti, inputs[0], axes, keep_dims)

    def convert_sym_size(self, node: torch.fx.Node, inputs):
        if IS_DYNAMIC:
            return None
        else: 
            if torch.__version__ > "2.4.2":
                from torch._subclasses.fake_tensor import unset_fake_temporarily
                no_fake_context = unset_fake_temporarily
            else:
                from torch.fx.experimental.proxy_tensor import (  # type: ignore[attr-defined] 
                    maybe_disable_fake_tensor_mode,
                )
                no_fake_context = maybe_disable_fake_tensor_mode

            with no_fake_context():
                # The torch.fx.Node does not have a "value" attribute
                # We rely on value until we can adopt a solution based on existing Node args
                # See https://github.ibm.com/IBM/torch_sendnn/pull/113#issuecomment-95086185
                sendnn_node = self.gb.ConstInput(node.name, 
                    convert_to_sendnn_tensor(torch.tensor(node.value, dtype=torch.int64)))  # type: ignore[attr-defined]
            return sendnn_node

    def convert_t(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        dim0 = 0
        dim1 = 1
        if str(node.target) == "aten.transpose.int":
            dim0 = node.args[1]
            dim1 = node.args[2]
        return self.gb.Transpose(node.name, ti, inputs[0], dim0, dim1)

    def convert_transpose(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        perm = convert_shape(node.args[1])
        return self.gb.Transpose(node.name, ti, inputs[0], perm, False)

    def convert_tril(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        diagonal = 0
        if len(node.args) == 2:
            diagonal = node.args[1]
        return self.gb.Trilu(node.name, ti, inputs[0], diagonal)

    def convert_where(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.Where3(node.name, ti, inputs[0], inputs[1], inputs[2])

    def convert_zeros(self, node, inputs):
        dt = convert_data_type(node)
        shape = convert_shape(node)
        layout = convert_layout(node.shape)
        ti = sendnn.TensorInfo(dt, shape, layout)
        return self.gb.Zeros(node.name, ti)

    fn_map = {
        "<built-in function getitem>": convert_get_item,
        "aten.abs.default": convert_singular_function(sendnn.GraphBuilder.Abs),
        "aten.add.Tensor": convert_binary_function(sendnn.GraphBuilder.Add),
        "aten.addmm.default": convert_addmm,
        "aten.arange.default": convert_arange,
        "aten.arange.start": convert_arange,
        "aten.avg_pool2d.default": convert_avg_pooling(sendnn.GraphBuilder.AvgPooling),
        "aten.bitwise_not.default": convert_singular_function(sendnn.GraphBuilder.LogicalNot),
        "aten.bmm.default": convert_matmul(sendnn.GraphBuilder.BatchMatMul),  # memory_format
        "aten.cat.default": convert_cat,
        "aten.clamp.default": convert_clamp,
        "aten.clamp_min.default": convert_clamp,
        "aten.clone.default": convert_singular_function(sendnn.GraphBuilder.Identity),
        "aten.convolution.default": convert_convolution,
        "aten.cumsum.default": convert_cumsum,
        "aten.detach.default": convert_singular_function(sendnn.GraphBuilder.Identity),
        "aten.div.Scalar": convert_binary_function(sendnn.GraphBuilder.RealDiv),
        "aten.div.Tensor": convert_binary_function(sendnn.GraphBuilder.RealDiv),
        "aten.native_dropout.default": convert_dropout,
        "aten.native_dropout_backward.default": convert_dropout_backward,
        "aten.embedding_dense_backward.default": convert_embedding_backward,
        "aten.full.default": convert_full,
        "aten.eq.Scalar": convert_binary_function(sendnn.GraphBuilder.Equal),
        "aten.eq.Tensor": convert_binary_function(sendnn.GraphBuilder.Equal),
        "aten.embedding.default": convert_embedding,
        "aten.expand.default": convert_singular_function(sendnn.GraphBuilder.BroadcastTo),
        "aten.exp.default": convert_singular_function(sendnn.GraphBuilder.Exp),
        "aten.ge.Scalar": convert_greater_equal,
        "aten.ge.Tensor": convert_greater_equal,
        "aten.gelu.default": convert_gelu,
        "aten.gelu_backward.default": convert_gelu_backward,
        "aten.lift_fresh_copy.default": convert_singular_function(sendnn.GraphBuilder.LiftFreshCopy),
        "aten.linalg_vector_norm.default": convert_norm,
        "aten.log.default": convert_singular_function(sendnn.GraphBuilder.Log),
        "aten._log_softmax.default": convert_log_softmax,
        "aten._log_softmax_backward_data.default": convert_log_softmax_backward,
        "aten.logical_not.default": convert_singular_function(sendnn.GraphBuilder.LogicalNot),
        "aten.lt.Tensor": convert_binary_function(sendnn.GraphBuilder.Less),
        "aten.index.Tensor": convert_index,
        "aten.max_pool2d_with_indices.default": convert_pooling(sendnn.GraphBuilder.MaxPooling),
        "aten.maximum.default": convert_binary_function(sendnn.GraphBuilder.Maximum),
        "aten.mean.dim": convert_reduce_mean,
        "aten.mish.default": convert_singular_function(sendnn.GraphBuilder.Mish),
        "aten.mm.default": convert_matmul(sendnn.GraphBuilder.MatMul),
        "aten.mul.Scalar": convert_binary_function(sendnn.GraphBuilder.Multiply),
        "aten.mul.Tensor": convert_binary_function(sendnn.GraphBuilder.Multiply),
        "aten._native_batch_norm_legit_functional.default": convert_batch_norm,
        "aten._native_batch_norm_legit_no_training.default": convert_batch_norm,
        "aten.native_layer_norm.default": convert_layer_norm,
        "aten.native_layer_norm_backward.default": convert_layer_norm_backward,
        "aten.ne.Scalar": convert_binary_function(sendnn.GraphBuilder.NotEqual),
        "aten.neg.default": convert_singular_function(sendnn.GraphBuilder.Negative),
        "aten.new_empty.default": convert_new_empty,
        "aten.new_full.default": convert_full,
        "aten.new_ones.default": convert_new_ones,
        "aten.new_zeros.default": convert_new_zeros,
        "aten.nll_loss_forward.default": convert_nll_loss,
        "aten.nll_loss_backward.default": convert_nll_loss_backward,
        "aten.nll_loss2d_forward.default": convert_nll_loss,
        "aten.nll_loss2d_backward.default": convert_nll_loss_backward,
        "aten.ones.default": convert_ones,
        "aten.permute.default": convert_transpose,
        "aten.pow.Scalar": convert_pow,
        "aten.pow.Tensor_Scalar": convert_pow,
        "aten.pow.Tensor_Tensor": convert_pow,
        "aten.reciprocal.default": convert_singular_function(sendnn.GraphBuilder.Reciprocal),
        "aten.relu.default": convert_singular_function(sendnn.GraphBuilder.Relu),
        "aten.repeat.default": convert_repeat,
        "aten.rsqrt.default": convert_singular_function(sendnn.GraphBuilder.Rsqrt),
        "aten.scalar_tensor.default": convert_scalar_tensor,
        "aten.slice.Tensor": convert_slice,
        "aten.slice_backward.default": convert_slice_backward,
        "aten.select_backward.default": convert_select_backward,
        "aten.sigmoid.default": convert_singular_function(sendnn.GraphBuilder.Sigmoid),
        "aten.silu.default": convert_singular_function(sendnn.GraphBuilder.Silu),
        "sendnn.silu_backward.default": convert_silu_backward,
        "aten._safe_softmax.default": convert_softmax,
        "aten._softmax.default": convert_softmax,
        "aten._softmax_backward_data.default": convert_softmax_backward,
        "aten.split.Tensor": convert_split,
        "aten.split_with_sizes.default": convert_split,
        "aten.sqrt.default": convert_singular_function(sendnn.GraphBuilder.Sqrt),
        "aten.squeeze.dim": convert_squeeze,
        "aten.stack.default": convert_stack,
        "aten.sub.Tensor": convert_binary_function(sendnn.GraphBuilder.Subtract),
        "aten.sum.dim_IntList": convert_sum,
        "aten.sym_size.int": convert_sym_size,
        "aten.t.default": convert_t,  # dtype, device
        "aten.tanh.default": convert_singular_function(sendnn.GraphBuilder.Tanh),
        "aten.tanh_backward.default": convert_binary_function(sendnn.GraphBuilder.TanhBackward),
        "aten._to_copy.default": convert_singular_function(sendnn.GraphBuilder.Identity),
        "aten.transpose.int": convert_t,
        "aten.tanh.Tensor": convert_singular_function(sendnn.GraphBuilder.Tanh),
        "aten.tril.default": convert_tril,
        "aten._unsafe_view.default": convert_reshape,
        "aten.unsqueeze.default": convert_expand_dims,
        "aten.upsample_nearest2d.default": convert_resize_nearest,
        "sendnn.upsample_nearest2d.default": convert_resize_nearest_sendnn,
        "aten.view.default": convert_reshape,
        "aten.where.self": convert_where,
        "aten.zeros.default": convert_zeros,
        "aten.zeros_like.default": convert_singular_function(sendnn.GraphBuilder.ZerosLike),
        "c10d_functional.all_reduce.default": convert_all_reduce,
        "_c10d_functional.all_reduce.default": convert_all_reduce_native,
        "c10d_functional.all_gather_into_tensor.default": convert_all_gather,
        "_c10d_functional.all_gather_into_tensor.default": convert_all_gather_native,
        "_c10d_functional.reduce_scatter_tensor.default": convert_reduce_scatter_native,
        "c10d_functional.wait_tensor.default": convert_wait_tensor,
        "_c10d_functional.wait_tensor.default": convert_wait_tensor,
        "autogptq_gemm.exv2_i4f16_fxinputs_aiu.default": convert_batch_matmul_w4a16,
        "gptq_gemm.i4f16_fxinputs_aiu.default": convert_batch_matmul_w4a16,
        "fms_mo.i8i8_aiu.default": convert_batch_matmul_w8a8,
    }

    def convert_call_module(self, node):
        raise RuntimeError("CallModule is not supported")

    def convert_call_function(self, node: torch.fx.Node):

        inputs = self.find_inputs(node)

        conv_fn = self.fn_map.get(str(node.target), FxToSenDnn.convert_unknown)
        return conv_fn(self, node, inputs)

    def convert_output(self, node):
        outputs = []
        out_idx = 0
        for outp in node.args[0]:
            if outp is not None:
                outputs.append(self.gb.PrimaryOutput(node.name + "_" + str(out_idx), self.sendnn_nodes[outp.name]))
                out_idx = out_idx + 1
        return outputs

    def convert_node(self, node):
        if node.op == "placeholder":
            return self.convert_placeholder(node)
        if node.op == "call_function":
            return self.convert_call_function(node)
        if node.op == "call_module":
            return self.convert_call_module(node)
        if node.op == "output":
            return self.convert_output(node)
        if node.op == "get_attr":
            return self.convert_getattr(node)

    def convert_graph(self, graph):
        for node in graph.nodes:
            sen_node = self.convert_node(node)
            if not sen_node:
                continue
            if isinstance(sen_node, list) and isinstance(sen_node[0], list):
                for e in sen_node:
                    self.sendnn_nodes[e[0]] = e[1]
            else:
                self.sendnn_nodes[node.name] = sen_node
        g = sendnn.Graph()
        status = self.gb.Finalize(g)
        if not status.IsOk():
            raise RuntimeError(f"GraphBuilder Finalize failed: {status}")
        log.info("G1: %s", g)
        return g


class GraphLoaderOp:
    def __init__(self, graph, ori_gm):
        self.gl = sendnn.GraphLoader("sen0")
        
        s = self.gl.LoadGraph(graph, False)
        if not s.IsOk():
            raise RuntimeError(f"GraphLoader LoadGraph failed: {s}")
        
        s = self.gl.CompileGraph()
        if not s.IsOk():
            raise RuntimeError(f"GraphLoader CompileGraph failed: {s}")

        s = self.gl.ParseGraph()
        if not s.IsOk():
            raise RuntimeError(f"GraphLoader ParseGraph failed: {s}")

        self.ori_gm = ori_gm
        self.fx_graph = torch.fx.Graph()
        log.info("FX Graph: " + str(self.fx_graph))
        self.sn_idx = 0
        self.sn_num = 0
        self.remove_prepare_model = False
        self.remove_device_init = False
        self.sn_op = functools.update_wrapper(functools.partial(self.sendnn_super_node_op), self.sendnn_super_node_op)
        self.ori_fx_nodes = {}
        self.fx_nodes = {}

    def sendnn_super_node_op(self, *inputs):
        pi_idx = 0
        pi_tensors = []
        tkv = 1
        for inp in inputs:
            if isinstance(inp, int):
                pi_tensors.append(np.array([inp]))
                tkv = inp
            else:
                pi_tensors.append(inp.numpy())
            pi_idx = pi_idx + 1

        po_idx = 0
        outputs = []
        # TODO: infer output shape for general symbolic shape, or move output allocation down the stack
        # TODO: is there a way to not allocate dummy kv cache?
        tkv += 1
        for sendnn_po in self.gl.GetOutputs(self.sn_idx):
            po_shape = sendnn_po.Shape().Dims()
            po_shape_int = []
            for d in po_shape:
                if isinstance(d, int):
                    po_shape_int.append(d)
                else:
                    po_shape_int.append(tkv)
            po_dt = convert_from_sendnn_data_type(sendnn_po.DataType())
            out = torch.zeros(()).new_empty(po_shape_int, dtype=po_dt).numpy()
            outputs.append(out)
            po_idx = po_idx + 1
        status = sendnn.Predict(self.gl, outputs, pi_tensors, self.sn_idx)
        if not status.IsOk():
            raise RuntimeError(f"sendnn Predict failed: {status}")
        self.sn_idx += 1
        if self.sn_idx == self.sn_num:
            if self.remove_prepare_model:
                self.sn_idx = 2
            elif self.remove_device_init:
                self.sn_idx = 1
            else:
                self.sn_idx = 0
        
        if not outputs:
            return None
        elif len(outputs) == 1:
            return torch.as_tensor(outputs[0])
        else:
            return [torch.as_tensor(x) for x in outputs]

    def is_add_getitem(self, node : sendnn.Node) -> bool:
        op = node.Fn()
        # TODO: check if any missing ops
        # ops that may have 1 or multiple outputs
        if (op == opcodes.Split or 
            op == opcodes.SenSuperNodeV2 or
            op == opcodes.AvgPool):
            if len(node.Successors()) > 1:
                return True
            else:
                return False
        # ops that have multiple outputs in fx, but
        # have 1 output in sendnn for inference and multiple outputs for training
        # TODO: change to have the same outputs as fx
        elif (op == opcodes.FusedBatchNorm or
              op == opcodes.LayerNorm or
              op == opcodes.MaxPool):
            return True
        elif len(node.Successors()) > 1:
            return True
        else:
            return False

    def is_add_wait(self, node : sendnn.Node) -> bool:
        op = node.Fn()
        if (op == opcodes.AllReduceSum or
            op == opcodes.AllGather):
            return True
        else:
            return False

    def convert_super_node(self, node):
        args = []
        for edge in node.Predecessors():
            src_node = edge.Source().Node()
            src_node_name = src_node.Name()
            fx_src_node = self.fx_nodes[src_node_name]
            
            # step to getitem if src node has multiple outputs
            # step to wait_tensor if communication nodes
            if self.is_add_getitem(src_node):
                src_idx = edge.Source().Index()
                for idx in range(src_idx + 1):
                    if len(src_node.Successors()[idx].Sinks()) == 0:
                        continue
                    fx_src_node = fx_src_node.next
            elif self.is_add_wait(src_node):
                fx_src_node = fx_src_node.next

            # strides will be handled natively in the stack as discussed. For now,
            # if not contiguous, add clone 
            # TODO: there could be corner cases
            is_contiguous = getattr(fx_src_node, "is_contiguous", True)
            if not is_contiguous:
                clone_args = (fx_src_node,)
                clone_kwargs = {'memory_format': torch.contiguous_format}
                fx_src_node = self.fx_graph.create_node("call_function", aten.clone.default, clone_args, clone_kwargs)
                
            args.append(fx_src_node)

        args_t = tuple(args)
        kwargs = {}  # dict
        super_node = self.fx_graph.create_node("call_function", self.sn_op, args_t, kwargs, node.Name())
        node_name = node.Name()
        self.fx_nodes[node_name] = super_node
        num_outputs = len(node.Successors())
        if num_outputs > 1:
            for idx in range(num_outputs):
                args_getitem = []
                args_getitem.append(super_node)
                args_getitem.append(idx)
                args_getitem_t = tuple(args_getitem)
                self.fx_graph.call_function(operator.getitem, args_getitem_t)

    def convert_lift_fresh_copy(self, node):
        # TODO: check if lift_fresh_copy always comes after get_attr
        getattr_target = node.Predecessors()[0].Source().Node().Name()
        getattr_node = self.fx_graph.create_node("get_attr", getattr_target)
        node_name = node.Name()
        ori_fx_node = self.ori_fx_nodes[node_name]
        target = ori_fx_node.target
        args = list(ori_fx_node.args)
        kwargs = ori_fx_node.kwargs
        args[0] = getattr_node
        args_t = tuple(args)
        new_node = self.fx_graph.create_node("call_function", target, args_t, kwargs, node_name)
        self.fx_nodes[node_name] = new_node

    def convert_new_full(self, node):
        node_name = node.Name()
        ori_fx_node = self.ori_fx_nodes[node_name]
        target = ori_fx_node.target
        args = list(ori_fx_node.args)
        kwargs = ori_fx_node.kwargs
        args[0] = self.fx_nodes[args[0].name]
        args_t = tuple(args)
        new_node = self.fx_graph.create_node("call_function", target, args_t, kwargs, node_name)
        self.fx_nodes[node_name] = new_node

    def convert_sym_size(self, ori_fx_node):
        node_name = ori_fx_node.name
        target = ori_fx_node.target
        args = list(ori_fx_node.args)
        kwargs = ori_fx_node.kwargs
        args[0] = self.fx_nodes[args[0].name]
        args_t = tuple(args)
        new_node = self.fx_graph.call_function(target, args_t, kwargs, node_name)
        self.fx_nodes[node_name] = new_node
        return new_node

    # TODO: refactor
    def convert_fallback(self, node):
        node_name = node.Name()
        ori_fx_node = self.ori_fx_nodes[node_name]
        target = ori_fx_node.target
        if target == aten.lift_fresh_copy.default:
            self.convert_lift_fresh_copy(node)
        elif target == aten.new_full.default:
            self.convert_new_full(node)
        else:
            args = ori_fx_node.args
            kwargs = ori_fx_node.kwargs
            args = list(args)
            # refactor
            arg_idx = 0
            for arg in args:
                if type(arg) == torch.fx.immutable_collections.immutable_list:
                    args_list = []
                    list_idx = 0
                    for n in arg:
                        new_n = n
                        if type(n) == torch.fx.node.Node:
                            if n.target == aten.sym_size.int:
                                if n.name not in self.fx_nodes:
                                    new_n = self.convert_sym_size(n)
                                else:
                                    new_n = self.fx_nodes[n.name]
                        args_list.append(new_n)
                        list_idx += 1
                    args[arg_idx] = args_list
                arg_idx += 1

            for i, edge in enumerate(node.Predecessors()):
                src_node = edge.Source().Node()
                src_node_name = src_node.Name()
                if src_node.Fn() == opcodes.ConstInput:
                    if src_node_name.startswith("scalar_tensor"):
                        ori_src_node = self.ori_fx_nodes[src_node_name]
                        new_src_node = self.fx_graph.create_node("call_function", ori_src_node.target,
                                                              ori_src_node.args, ori_src_node.kwargs,
                                                              src_node_name)
                        args[i] = new_src_node
                    continue
                fx_src_node = self.fx_nodes[src_node_name]

                # step to getitem if src node has multiple outputs
                # step to wait_tensor if communication nodes
                if self.is_add_getitem(src_node):
                    src_idx = edge.Source().Index()
                    for idx in range(src_idx + 1):
                        if len(src_node.Successors()[idx].Sinks()) == 0:
                            continue
                        fx_src_node = fx_src_node.next
                elif self.is_add_wait(src_node):
                    fx_src_node = fx_src_node.next

                if (str(target) == "aten.cat.default" or
                    str(target) == "aten.stack.default"):
                    if i == 0:
                        args[0] = [fx_src_node]
                    else:
                        args[0].append(fx_src_node)
                elif str(target) == "aten._unsafe_index.Tensor":
                    if i==0:
                        args[0] = fx_src_node
                    else:
                        position = 0
                        # find not 'None' position in args[1]
                        if type(args[1]) == torch.fx.immutable_collections.immutable_list:
                            args[1] = list(args[1])
                        for n,v in enumerate(args[1]):
                            if v != None:
                                position += 1
                                if position == i:
                                    args[1][n] = fx_src_node
                                    break
                elif str(target) == "aten.index.Tensor":
                    if i == 1:
                        args[i] = [fx_src_node]
                    else:
                        args[i] = fx_src_node
                elif ((str(target) == "aten.native_layer_norm.default" and i > 0) or
                      (str(target) == "aten.native_layer_norm_backward.default" and i > 1)):
                    i += 1
                    args[i] = fx_src_node
                elif ((str(target) == "aten.nll_loss_backward.default" or 
                       str(target) == "aten.nll_loss2d_backward.default") and i == 3):
                    i = 6
                    args[i] = fx_src_node
                else:
                    args[i] = fx_src_node

            args = tuple(args)
            new_node = self.fx_graph.create_node("call_function", target, args, kwargs, node.Name())
            new_node.is_contiguous = getattr(ori_fx_node, "is_contiguous", True)
            self.fx_nodes[node_name] = new_node

            if self.is_add_getitem(node):
                num_outputs = len(ori_fx_node.users)
                getitem_node = ori_fx_node
                for _ in range(num_outputs):
                    getitem_node = getitem_node.next
                    is_contiguous = getattr(getitem_node, "is_contiguous", True)
                    getitem_args = list(getitem_node.args)
                    getitem_args[0] = new_node
                    getitem_args_t = tuple(getitem_args)
                    new_getitem_node = self.fx_graph.call_function(operator.getitem, getitem_args_t)
                    new_getitem_node.is_contiguous = is_contiguous
            elif self.is_add_wait(node):
                is_contiguous = getattr(ori_fx_node.next, "is_contiguous", True)
                args_wait = []
                args_wait.append(new_node)
                args_wait_t = tuple(args_wait)
                kwargs_wait = {}
                target_wait = torch.ops.c10d_functional.wait_tensor
                new_wait_node = self.fx_graph.create_node("call_function", target_wait, args_wait_t, kwargs_wait)
                new_wait_node.is_contiguous = is_contiguous

    def convert_output(self, output_ops):
        args = []
        for node in self.ori_gm.graph.nodes:
            if node.op == "output":
                ori_output = node
        out_idx = 0
        for inp in ori_output.args[0]:
            if inp is None:
                args.append(inp)
            else:
                node = output_ops[out_idx]
                out_idx = out_idx + 1
                for edge in node.Predecessors():
                    indexed_node = edge.Source()
                    src_node = indexed_node.Node()
                    src_node_name = src_node.Name()
                    fx_src_node = self.fx_nodes[src_node_name]
                    if self.is_add_getitem(src_node):
                        src_idx = edge.Source().Index()
                        for idx in range(src_idx + 1):
                            if len(src_node.Successors()[idx].Sinks()) == 0:
                                continue
                            fx_src_node = fx_src_node.next
                    elif self.is_add_wait(src_node):
                        fx_src_node = fx_src_node.next
                    args.append(fx_src_node)
        args_t = tuple(args)
        self.fx_graph.output(args_t)

    def graph(self) -> torch.fx.Graph:
        g2 = self.gl.GetG2()

        for fx_node in self.ori_gm.graph.nodes:
            self.ori_fx_nodes[fx_node.name] = fx_node

        # Inputs
        # the compiled fx graph needs to have the same number of inputs as original fx graph
        # dummy inputs might be optimized away in g2
        for ori_fx_node in self.ori_gm.graph.nodes:
            if ori_fx_node.op == "placeholder":
                inp_name = ori_fx_node.name
                fx_node = self.fx_graph.placeholder(inp_name)
                self.fx_nodes[inp_name] = fx_node
        # Computes
        for node in g2.compute_ops:
            if node.Fn() == opcodes.SenSuperNodeV2:
                self.convert_super_node(node)
                self.sn_num += 1
            else:
                self.convert_fallback(node)

        # clean_graph node
        self.fx_graph.create_node("call_function", clean_graph, (), {})

        # Outputs
        self.convert_output(g2.output_ops)
        
        if is_bwd_graph(self.ori_gm.graph):
            global model_inputs
            model_inputs.clear()

        self.fx_graph.lint()
        log.info("G2: %s", g2)
        log.info("Modified FX Graph: %s", self.fx_graph)
        return self.fx_graph

    def clean_graph(self):
        has_preparemodel = False
        has_deviceinit = False
        for node in self.fx_graph.nodes:
            if node.name == "PrepareModel":
                has_preparemodel = True
            elif node.name == "DeviceInit":
                has_deviceinit = True
        self.fx_graph.eliminate_dead_code()
        log.info("Modified FX Graph: %s", self.fx_graph)
        if has_preparemodel:
            self.remove_prepare_model = True
            self.sn_idx = 2
        elif has_deviceinit:
            self.remove_device_init = True
            self.sn_idx = 1
        return self.fx_graph

    def graph_module(self):
        return torch.fx.GraphModule(self.ori_gm, self.graph())


def create_empty_graph(aot_autograd_graph: torch.fx.Graph):
    """
    This function creates a temporary empty graph for a LazyHandle until the real compilation
    happens in deeptools so torch.compile() can finish execution properly.
    """
    new_graph = torch.fx.Graph()

    # need to retain module attributes

    # Set output_node to first node in graph until output node is found
    output_node = iter(aot_autograd_graph.nodes)
    # Grab all inputs and parameters of the model represented for the graph
    # Use the same loop ot grab the output node to save one iteration
    for node in aot_autograd_graph.nodes:
        if node.op == "placeholder":  # input to graph
            new_graph.placeholder(node.name)  # noqa F841
        elif node.op == "get_attr":  # parameter of the model in graph
            # getattr_node is not used for now
            new_graph.create_node("get_attr", node.target)  # noqa F841
        elif node.op == "output":  # the output of the graph
            output_node = node
    
    # Create empty allocations in cpu for all the outputs of the graph
    empty_nodes = []
    for node in output_node.args[0]:
        shape = []
        dtype = torch.float32
        if getattr(node, "shape", None):
            shape = node.shape
            dtype = node.dtype
        
        # Grab shape for the empty call, including dynamic shapes
        shape_list = list(shape)
        for idx, elem in enumerate(shape_list):
            if not isinstance(elem, int):
                if isinstance(elem, torch.fx.Node):
                    shape_list[idx] = elem.value  # type: ignore[attr-defined]
                if isinstance(elem, torch.SymInt):
                    # See [Note: Usage of SymInt.node._hint for dynamic shapes]
                    shape_list[idx] = elem.node._hint
        empty_args = (torch.Size(shape_list),)
        kwargs: dict[str, Any] = {'dtype': dtype}
        new_empty_node = new_graph.create_node("call_function", aten.empty, empty_args, kwargs)
        empty_nodes.append(new_empty_node)

    # Create a new output node with the empty allocations
    output_args = tuple(empty_nodes)
    new_graph.output(output_args)
    log.info("Empty graph: %s", new_graph)
    return new_graph


class LazyHandle(torch.fx.GraphModule):
    def __init__(
        self,
        aot_autograd_gm: GraphModule,
        fake_tensor_inputs: list[torch.Tensor],
        shape_env: ShapeEnv | None,
        is_warmup=False,
    ):
        self.propagate_shapes(aot_autograd_gm, fake_tensor_inputs, shape_env)
        # Create the placeholder empty graph until real compilation happens
        if shape_env is not None:
            with shape_env.suppress_guards():
                empty_graph = create_empty_graph(aot_autograd_gm.graph)
        else:
            empty_graph = create_empty_graph(aot_autograd_gm.graph)
        super().__init__(aot_autograd_gm, empty_graph)
        self.init_from_tracing_context()

        # When a torch.fx.GraphModule is copied, only metadata in self.meta is preserved
        self.meta["original_gm"] = aot_autograd_gm
        self.meta["shape_id"] = compute_shape_id(fake_tensor_inputs, {})

        self.meta["is_compiled"] = False
        self.meta["g1"] = None
        self.meta["g2"] = None
        self.meta["graph_key"] = None

        if use_aiu_cache and not is_warmup:
            # Try to find and load a compiled graph if it already exists and not in warmup mode
            self.meta["graph_key"] = cache.generate_graph_key(
                aot_autograd_gm, fake_tensor_inputs, self.meta["shape_id"], 1
            )
            self.load_graph_from_cache(self.meta["graph_key"])

        if use_aiu_cache and self.meta["is_compiled"]:
            # Convert placeholder nodes in FX graph
            self.convert_fx_graph(aot_autograd_gm, fake_tensor_inputs)
        
	# If cache is disabled or didn't hit... (is_compiled is set inside 
        # load_graph_from_cache if a hit happens)
        if not self.meta["is_compiled"]:
            self.create_sendnn_graph(aot_autograd_gm, fake_tensor_inputs, shape_env)
            # Save input metadata for later cache key computation
            self.meta["example_inputs"] = fake_tensor_inputs
        self.meta["gl_op"] = None

    def init_from_tracing_context(self):
        self.meta["is_grad_enabled"] = False
        self.meta["static_input_indices"] = []
        self.meta["is_optim"] = False

        # Grab relevant information for compiler from tracing context
        if tracing_context := TracingContext.get():
            params_flat = tracing_context.params_flat
            global_context = tracing_context.global_context
            grad_enabled = global_context.global_state.get("grad_enabled", None)
            self.meta["is_grad_enabled"] = grad_enabled[1] if grad_enabled else False
            if torch.__version__ > "2.4.2":
                if tracing_context.fw_metadata != None:
                    self.meta["static_input_indices"] = (
                        tracing_context.fw_metadata.static_input_indices
                )
            else:
                self.meta["static_input_indices"] = [i for i in range(len(params_flat))]
            for module in tracing_context.module_context.nn_modules:
                if "self___param_groups" in module:
                    self.meta["is_optim"] = True
                    break

    def create_sendnn_graph(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
        shape_env: ShapeEnv | None,
    ):
        fx_to_sendnn_converter = FxToSenDnn(
            gm,
            example_inputs,
            self.meta["static_input_indices"],
            self.meta["is_grad_enabled"],
            self.meta["is_optim"],
	)
        with shape_env.suppress_guards() if shape_env is not None else nullcontext():
            self.meta["g1"] = fx_to_sendnn_converter.convert_graph(gm.graph) 
        self.meta["g2"] = self.meta["g1"]	
    
    def convert_fx_graph(
        self, 
        gm: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
	# Convert placeholder nodes in FX graph 
	# Adapted from FxToSenDnn.convert_placeholder
        mi_idx = 0
        pi_idx = 0
        args_idx = 0 
        args_iter = iter(example_inputs)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                inp = next(args_iter)
                if not isinstance(inp, torch.SymInt):
                    args_idx += 1
                    if not self.meta["is_grad_enabled"]:
                        idx = args_idx - 1
                        if idx in self.meta["static_input_indices"]:
                            node.name = "mi_" + str(mi_idx)
                            mi_idx += 1
                        else:
                            node.name = "pi_" + str(pi_idx)
                            pi_idx += 1
            else:
                args_idx += 1

    def propagate_shapes(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor], shape_env: ShapeEnv | None):
        shape_prop = MyShapeProp(gm, shape_env=shape_env)
        shape_prop.propagate(*example_inputs)

    def update_graph(self):
        self.meta["gl_op"] = GraphLoaderOp(self.meta["g2"], self.meta["original_gm"])
        self.meta["g2"] = self.meta["gl_op"].gl.GetG2()

        # Store compiled g2 from gl_op if it hasn't been loaded from cache
        if use_aiu_cache and not self.meta["is_compiled"]:
            cache.store_compiled_graph(self.meta["g2"], self.meta["graph_key"])
        self.meta["is_compiled"] = True
        self.graph = self.meta["gl_op"].graph()

    def load_graph_from_cache(self, cache_key: str):
        # Try to load the relevant g2 graph from disk
        cached_graph_path = cache.get_graph_path(cache_key)
        
        # If there's a cache hit load from disk
        if cached_graph_path is not None:
            log.info("Cache hit, loading G2 from %s", cached_graph_path)
            event_counter["torch_sendnn"]["cache_hits"] += 1
            self.meta["g1"] = None
            self.meta["g2"] = sendnn.Graph()    
            cache.load_compiled_graph(self.meta["g2"], cached_graph_path)
            self.meta["is_compiled"] = True
        else:
            event_counter["torch_sendnn"]["cache_misses"] +=1 

    def clean_graph(self):
        self.graph = self.meta["gl_op"].clean_graph()


# This global var holds all the handles to graphs our torch.compile() backend has seen 
lazy_handles: list[LazyHandle] = []

#====================================================
# Modification for repro code and other modules
#====================================================
_preserve_lazy_handle = False
def clean_graph():
    global lazy_handles, _preserve_lazy_handle
    if not _preserve_lazy_handle:
        lazy_handles[0].clean_graph()
        lazy_handles.pop(0)

def release_lazyhandle():
    global _preserve_lazy_handle
    _preserve_lazy_handle = False
    clean_graph()

def preserve_lazyhandle():
    global _preserve_lazy_handle
    _preserve_lazy_handle = True
#====================================================

def compute_shape_id(args, kwargs) -> tuple[int]:
    shape_id = []
    for arg in list(args) + list(kwargs.values()):
        if isinstance(arg, torch.SymInt):
            # See [Note: Usage of SymInt.node._hint for dynamic shapes]
            value = arg.node._hint
            shape_id.append(value)
        elif isinstance(arg, int):
            shape_id.append(arg)
    return tuple(shape_id)


# [Note: How do dynamic shapes work in torch dynamo + torch_sendnn]
# When compiling static shapes, torch dynamo will trace a new graph for each combination of input shapes
# This traced graph will then be sent to the "sendnn" backend, which will create a LazyHandle. 
# A LazyHandle is a Callable construct that gets returned by the "sendnn" backend to the Dynamo dispatcher,
# which will then gate this particular LazyHandle behind a set of guards to ensure the code being run is
# always sound (sound meaning the outputs will match the eager behavior).
# Now, the process diverges based on the _warmup_mode global variable:
#   1. if _warmup_mode = False, the traced graph will be immediately compiled by deeptools, and the compiled
#      graph will be made part of the LazyHandle, which is then returned back to dynamo and executed.
#   2. if _warmup_mode = True, the LazyHandle gets saved to a list of LazyHandles to be compiled, and as soon
#      as the warmup_mode context exits, the update_lazyhandle() function is called, which calls deeptools
#      with every single graph at the same time, allowing for deeptools to compile less graphs (right now,
#      64 graphs with increasing sequence lengths get a single compiled artifact)
# 
#  But what happens with dynamic shapes? First, when dynamic shapes are detected by dynamo in a graph, all 
#  tensors with the dynamic shapes will get marked by dynamo as having SymInt shapes, and the shapes themselves
#  will become part of the "fake_tensor_inputs" that dynamo sends to the "sendnn" backend. This allows us to 
#  easily know if a graph has dynamic shapes, with a simple type test. Unlike static tracing, though, dynamo will
#  not trace more graphs with the same dynamic shapes, which means the "sendnn" backend will only get called once.
#  This means a few conditions for our backends:
#    1. If we have dynamic shapes, we currently need to ensure _warmup_mode = True, as otherwise we will
#       immediately compile and lose any possible optimization on the deeptools side.
#    2. We need a way to trace new graphs without getting new calls to the "sendnn" backend, as future calls to
#       this graph with varying shapes will not go through the "sendnn" backend anymore, instead going to the
#       Callable we return on this first call.
# This is where the new SpyreGraphExecutor comes in. This new Callable acts like a superpowered
# LazyHandle, as it saves the dynamic graph and the original inputs with the dynamic shapes. Its workflow is the 
# following:
#    1. The first time a dynamic shapes graph is received, "sendnn" will get called and will check for dynamic
#       shapes. If they exist and we are in warmup mode, it will create a SpyreGraphExecutor object.
#    2. This SpyreGraphExecutor is returned as the Callable that torch dynamo will call whenever
#       the dynamic graph has a hit on its cache. Whenever that hit happens, here's what the __call__ function does:
#       First, we compute a hash of the real shapes and check if we have seen this variation yet. Then:
#       a) If *we have not seen it* AND *we have not seen a single variation yet*, we change all the original 
#          dynamic fake_tensor_inputs to be static, and create a new LazyHandle for it, which we also save as a
#          reference for future calls and add to the list of graphs to be compiled.
#       b) If *we have not seen it* AND *we have seen some variations already*, we create a copy of the reference
#          LazyHandle, re-static the dynamic inputs with the new shapes, add the LazyHandle to the list of graphs
#          to be compiled.
#          In both cases, after all this, we add this to the hash map of shapes seen to LazyHandles
#       c) If *we have seen it*, we simply grab the LazyHandle back from the shapes_to_lh_map map
#       Finally, we call the correct LazyHandle with the arguments passed to the __call__ function.
class SpyreGraphExecutor:
    def __init__(
        self,
        traced_graph_module: GraphModule,
        fake_tensor_inputs: list[torch.Tensor],
        shape_env: ShapeEnv | None,
        is_warmup=False,
    ):
        self.is_warmup = is_warmup
        self.traced_graph_module = traced_graph_module
        self.fake_tensor_inputs = fake_tensor_inputs
        self.shape_env = shape_env
        self.shapes_to_lh_map = {}
        self.orig_lh = self.__get_original_lazy_handle()

    def __replace_dynamic_inputs(self, args, kwargs):
        static_graph_inputs = []
        # fake_tensor_inputs and full_arg_list have exactly the same elements, in the same order
        full_arg_list = list(args) + list(kwargs.values())
        for arg_idx, fake_tensor_input in enumerate(self.fake_tensor_inputs):
            if isinstance(fake_tensor_input, torch.Tensor):
                # 1. Grab the same tensor from real args to pick the real dimensions
                real_tensor = full_arg_list[arg_idx]

                if not isinstance(real_tensor, torch.Tensor):
                    log.warning(f"Expected real_tensor to be an instance of torch.Tensor, instead got {type(real_tensor)}")

                # If the fake tensor is dynamic in any dimension
                if any([isinstance(s, torch.SymInt) for s in fake_tensor_input.shape]):
                    # 2. Create a new fake tensor of correct shape
                    static_graph_inputs.append(
                        fake_tensor_input.new_empty(real_tensor.shape)
                    )
                else:
                    static_graph_inputs.append(fake_tensor_input)
            else:
                # Deal with the dynamic shapes themselves
                if isinstance(fake_tensor_input, torch.SymInt):
                    real_int = full_arg_list[arg_idx]

                    if not isinstance(real_int, (int, torch.SymInt)):
                        log.warning(f"Expected real_int to be an instance of int or torch.SymInt, instead got {type(real_int)}")

                    if isinstance(real_int, int):
                        real_hint = real_int
                    else:
                        # See [Note: Usage of SymInt.node._hint for dynamic shapes]
                        real_hint = real_int.node._hint
                    fake_tensor_input.node._hint = real_hint
                # Otherwise, just save original
                static_graph_inputs.append(fake_tensor_input)
        return static_graph_inputs

    def __get_inputs_metadata(self, example_inputs):
        inputs_metadata = []
        for example_input in example_inputs:
            if getattr(example_input, "device", None) is not None:
                inputs_metadata.append(example_input.to(device="meta"))
            else:
                inputs_metadata.append(example_input)
        return inputs_metadata

    def __get_original_lazy_handle(self):
        global lazy_handles
        if self.shape_env is None:
            fake_tensor_inputs = self.fake_tensor_inputs
        else:
            fake_tensor_inputs = self.__replace_dynamic_inputs(
                self.fake_tensor_inputs, {}
            )

        orig_lh = LazyHandle(
            self.traced_graph_module, fake_tensor_inputs, self.shape_env, self.is_warmup
        )
        # add to lazy_handles for static shape; for dynamic shape, add in __call__
        if self.shape_env is None:
            lazy_handles.append(orig_lh)
        return orig_lh

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global lazy_handles
        # we have static shapes
        if self.shape_env is None:
            lh = self.orig_lh
        else:
            # Compute a unique id for this combination of dynamic shape hints
            shape_id = compute_shape_id(args, kwargs)
            if shape_id not in self.shapes_to_lh_map:
                # Create a new lazy handle if it's not there yet for this shape_id
                static_graph_inputs = self.__replace_dynamic_inputs(args, kwargs)

                # Optimize the copy by removing heavy and unnecessary objects from deepcopy
                orig_g1 = self.orig_lh.meta["g1"]
                orig_g2 = self.orig_lh.meta["g2"]
                orig_graph_key = self.orig_lh.meta["graph_key"]
                orig_example_inputs = self.orig_lh.meta["example_inputs"]

                self.orig_lh.meta["g1"] = None
                self.orig_lh.meta["g2"] = None
                self.orig_lh.meta["graph_key"] = None
                self.orig_lh.meta["original_gm"] = None
                self.orig_lh.meta["example_inputs"] = None
                lh = copy.deepcopy(self.orig_lh)
                self.orig_lh.meta["g1"] = orig_g1
                self.orig_lh.meta["g2"] = orig_g2
                self.orig_lh.meta["graph_key"] = orig_graph_key
                self.orig_lh.meta["original_gm"] = self.traced_graph_module
                self.orig_lh.meta["example_inputs"] = orig_example_inputs

                # Change relevant metadata for new lazy handle
                lh.meta["original_gm"] = self.traced_graph_module
                lh.meta["is_compiled"] = False
                lh.meta["shape_id"] = shape_id
                lh.meta["example_inputs"] = self.__get_inputs_metadata(
                    static_graph_inputs
                )
                lh.propagate_shapes(self.traced_graph_module, static_graph_inputs, self.shape_env)

                if use_aiu_cache and not self.is_warmup:
                    lh.meta["graph_key"] = cache.generate_graph_key(
                        self.traced_graph_module,
                        static_graph_inputs,
                        shape_id,
                        1,
                    )
                    lh.load_graph_from_cache(lh.meta["graph_key"])
                if use_aiu_cache and lh.meta["is_compiled"]:
                    lh.convert_fx_graph(lh.meta["original_gm"], lh.meta["example_inputs"])	
                if not lh.meta["is_compiled"]:
                    lh.create_sendnn_graph(
                        self.traced_graph_module, static_graph_inputs, self.shape_env
                    )
                    lh.graph = create_empty_graph(self.traced_graph_module.graph)

                lazy_handles.append(lh)
                self.shapes_to_lh_map[shape_id] = lh
            lh = self.shapes_to_lh_map[shape_id]
        return lh(*args, **kwargs)

def torch_sendnn_decoder(gm: GraphModule, fake_tensor_inputs: list[torch.Tensor]):
    if IS_DYNAMIC:
        global lazy_handles
        shape_env = None
        for fake_input in fake_tensor_inputs:
            if isinstance(fake_input, torch.SymInt):
                log.info("We have dynamic shapes!")
                shape_env = fake_input.node.shape_env
                break
        lh = LazyHandle(gm, fake_tensor_inputs, shape_env)
        lazy_handles.append(lh)
        return lh
    else:
        # check for dynamic shapes and if there are some, add a guard for sequence length
        log.info("Checking for dynamic shapes...")
        shape_env = None
        for fake_input in fake_tensor_inputs:
            if isinstance(fake_input, torch.SymInt):
                log.info("We have dynamic shapes!")
                shape_env = fake_input.node.shape_env
                break

        return SpyreGraphExecutor(gm, fake_tensor_inputs, shape_env, is_warmup=True)


def update_lazyhandle():
    global lazy_handles

    # First, count how many graphs we are compiling and compute the
    # cache keys for all graphs taking that into account
    num_compiled_graphs = len(lazy_handles)
    g1s = []
    for lh in lazy_handles:
        # If we are using the cache, we compute the key and try to
        # load the graph from cache
        if use_aiu_cache:
            lh.meta["graph_key"] = cache.generate_graph_key(
                lh.meta["original_gm"],
                lh.meta["example_inputs"],
                lh.meta["shape_id"],
                num_compiled_graphs,
            )
            lh.load_graph_from_cache(lh.meta["graph_key"])
        if lh.meta["is_compiled"]:
            # Successful cache load
            lh.update_graph()
        else:
            # In all other cases, we need to compile G1 graph
            g1s.append(lh.meta["g1"])

    # Compile necessary graphs together for optimization
    # This can only happen for cache misses or new graphs
    g2s = []
    if len(g1s) > 0:
        g2s = sendnn.CompileGraphs(g1s)

        for lh, g2 in zip(lazy_handles, g2s):
            lh.meta["g2"] = g2
            lh.update_graph()

_warmup_mode = False

def set_warmup_mode(status):
    global _warmup_mode
    _warmup_mode = status

def get_warmup_mode():
    global _warmup_mode
    return _warmup_mode

def torch_sendnn(gm, fake_tensor_inputs):
    global lazy_handles, _warmup_mode
    if IS_DYNAMIC:
        shape_env = None
        for fake_input in fake_tensor_inputs:
            if isinstance(fake_input, torch.SymInt):
                log.info("We have dynamic shapes!")
                shape_env = fake_input.node.shape_env
                break
        lh = LazyHandle(gm, fake_tensor_inputs, shape_env)
        lazy_handles.append(lh)
        if not _warmup_mode:
            lh.UpdateGraph()
        return lh
    else:
        # check for dynamic shapes and if there are some, add a guard for sequence length
        log.info("Checking for dynamic shapes...")
        shape_env = None
        for fake_input in fake_tensor_inputs:
            if isinstance(fake_input, torch.SymInt):
                log.info("We have dynamic shapes!")
                shape_env = fake_input.node.shape_env
                break
        graph_executor = SpyreGraphExecutor(gm, fake_tensor_inputs, shape_env, _warmup_mode)
        if not _warmup_mode:
            graph_executor.orig_lh.update_graph()
        return graph_executor


class warmup_mode:
    def __init__(self, enabled: bool = True):
        global _warmup_mode
        self._old_mode = _warmup_mode
        _warmup_mode = enabled

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _warmup_mode
        update_lazyhandle()
        _warmup_mode = self._old_mode

use_aiu_cache = os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0") == "1"
if use_aiu_cache:
    cache = SpyreGraphCache()

# TODO: Add read only feature
is_read_only_cache = os.environ.get('TORCH_SENDNN_CACHE_READ_ONLY', "0") == "1"

IS_DYNAMIC = False

def sendnn_backend(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], **kwargs):
    if options := kwargs.get('options', None):
        global IS_DYNAMIC
        IS_DYNAMIC = options.get('sendnn.dynamic', False)

    with prevent_op_decomp():
        return aot_autograd(
            fw_compiler=torch_sendnn,
            bw_compiler=torch_sendnn,
            inference_compiler=torch_sendnn,
            decompositions=sendnn_decompositions,
        )(graph_module, example_inputs, **kwargs)


# This is deprecated and to be removed. warmup_mode is to be used in conjunction with
# the sendnn_backend to support decoders
def sendnn_decoder_backend(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], **kwargs):
    if options := kwargs.get('options', None):
        global IS_DYNAMIC
        IS_DYNAMIC = options.get('sendnn.dynamic', False)

    log.warning("You're using a deprecated backend. Please use sendnn in conjuction with warmup_mode")
    with prevent_op_decomp():
        return aot_autograd(
            fw_compiler=torch_sendnn_decoder,
            bw_compiler=None,
            inference_compiler=torch_sendnn_decoder,
            decompositions=sendnn_decompositions,
        )(graph_module, example_inputs, **kwargs)
