import os
import torch

import re
from deepview.utils.model_handler import setup_model_handler
from sendnn import opcodes
from torch_sendnn.backends.sendnn_backend import __state
from torch_sendnn.conversion.conversion_utils import (
    shape_to_list,
    torch_datatype_to_sendnn,
)
import torch
node_ops_global_var=[]
node_ops_global_var_1=[]
filegen_count = 0

def get_all_ops(lazy_handle):
    all_ops = []
    g2 = lazy_handle.g2 if hasattr(lazy_handle,'g2') else lazy_handle.meta['g2']
    for op in g2.compute_ops:
        all_ops.append(op.Name())
    return all_ops

def sanitize_arg(arg):
    #pdb.set_trace()
    if isinstance(arg, list):
        return "[" + ", ".join([sanitize_arg(i) for i in arg]) + "]"

    if isinstance(arg, torch.fx.node.Node):
        if "float" in str(arg.meta["val"].dtype):
            return f"{arg.meta['val'].shape}$torch.rand({arg.meta['val'].shape})"
        elif "int" in str(arg.meta["val"].dtype):
            return f"{arg.meta['val'].shape}$torch.randint(1, {arg.meta['val'].shape})"
        elif "bool" in str(arg.meta["val"].dtype):
            return f"{arg.meta['val'].shape}$torch.rand({arg.meta['val'].shape}) < 0.9"
        else:
            print(f"Unhandled arg.dtype {arg.meta['val'].dtype}")
            return str(arg)

    return str(arg)

def get_nn_module(op: str):
    op_name = op.lower()

    # -----------------------------
    # Forward-only high-level ops
    # -----------------------------
    TORCH_OPS = {
        # arithmetic
        "add":               "torch.add",
        "add.tensor":        "torch.add",
        "add.scalar":        "torch.add",

        "sub":               "torch.sub",
        "sub.tensor":        "torch.sub",
        "sub.scalar":        "torch.sub",

        "mul":               "torch.mul",
        "mul.tensor":        "torch.mul",
        "mul.scalar":        "torch.mul",

        "eq":                "torch.eq",
        "eq.tensor":         "torch.eq",
        "eq.scalar":         "torch.eq",

        # shape
        "where":             "torch.where",
        "where.self":      "torch.where",
        "transpose":         "torch.transpose",
        "transpose.int":   "torch.transpose",
        "squeeze":           "torch.squeeze",
        "expand":            "lambda x, *s: x.expand(*s)",
        "view":              "lambda x, *s: x.view(*s)",

        "_unsafe_view":      "torch.ops.aten._unsafe_view",
        "slice.tensor":      "lambda x, dim, start, end: torch.narrow(x, dim, start, end - start)",

        # reduction
        "sum.dim_intlist":   "lambda x, dim, keepdim=False: torch.sum(x, dim=dim, keepdim=keepdim)",
        "mean.dim":          "lambda x, dim, keepdim=False: torch.mean(x, dim=dim, keepdim=keepdim)",

        # math
        "pow.tensor_scalar": "lambda x, y: torch.pow(x, y)",

        # copy/utility
        "_to_copy":          "torch.ops.aten._to_copy",
        "lift_fresh_copy":   "torch.ops.aten.lift_fresh_copy",
    }

    if op_name in TORCH_OPS:
        return TORCH_OPS[op_name], op_name

    # -----------------------------
    # nn.functional ops
    # -----------------------------
    FUNC_OPS = {
        "embedding":   "torch.nn.functional.embedding",
        "layer_norm":  "torch.nn.functional.layer_norm",
        "log_softmax": "torch.nn.functional.log_softmax",
        "softmax":     "torch.nn.functional.softmax",
        "silu":        "torch.nn.functional.silu",
    }

    if op_name in FUNC_OPS:
        return FUNC_OPS[op_name], op_name


    # -----------------------------
    # Aten forward ops
    # -----------------------------
    ATEN_FORWARD = {
        "nll_loss_forward": "torch.ops.aten.nll_loss_forward",
    }

    if op_name in ATEN_FORWARD:
        return ATEN_FORWARD[op_name], op_name

    # -----------------------------
    # Aten backward ops
    # -----------------------------
    ATEN_BACKWARD = {
        "nll_loss_backward":           "torch.ops.aten.nll_loss_backward",
        "native_layer_norm_backward":  "torch.ops.aten.native_layer_norm_backward",
        "gelu_backward":               "torch.ops.aten.gelu_backward",
        "tanh_backward":               "torch.ops.aten.tanh_backward",
        "select_backward":             "torch.ops.aten.select_backward",
        "slice_backward":              "torch.ops.aten.slice_backward",
    }

    if op_name in ATEN_BACKWARD:
        return ATEN_BACKWARD[op_name], op_name   # return callable

    # -----------------------------
    # Fallback: torch.<op>
    # -----------------------------
    if hasattr(torch, op_name):
        return "getattr(torch, \""+ op_name + "\")", op_name

    print(f"[WARN] Unsupported op: {op_name}")
    return None, op_name
def generate_reproduction(lazy_handle_id, node_name, target_name, args, yaml_gen):
    f_vars = []
    g_expr = []
    f_vars_1 = []
    g_expr_1 = []
    g_expr_3 = []
    global filegen_count
    for i in range(len(args)):
        f_vars.append(f"var{i}")
        f_vars_1.append(f"var{i}.cpu()")
        f_vars_1.append(f"var{i}.grad")
        modified_args = args[i].split('$')[0]
        if args[i].startswith("[") and "$" in args[i]:
            raw_input = args[i].strip("[]")
            pattern = r'(?:[^,(]|\([^)]*\))+'
            items = re.findall(pattern, raw_input)
            first_parts = []
            second_parts = []
            for item in items:
                if "$" in item:
                    a, b = item.split("$", 1)
                else:
                    a = b = item
                first_parts.append(a)
                second_parts.append(b)

            # Convert lists to string format
            modified_args = "[" + ",".join(first_parts) + "]"
            modified_args_1 = "[" + ",".join(second_parts) + "]"
            g_expr.append(f"var{i} = {modified_args_1}")
            t=eval(f"{modified_args_1}")
            g_expr_3.append(t)
        elif "$" in args[i]:
            modified_args_1 = args[i].split('$')[1]
            g_expr.append(f"var{i} = {modified_args_1}")
            t=eval(f"{modified_args_1}")
            g_expr_3.append(t)
        else:
            g_expr.append(f"var{i} = {args[i]}")
            g_expr_3.append(eval(args[i]))
        modified_args = modified_args.replace("torch.Size(","")
        modified_args = modified_args.replace(")","")
        g_expr_1.append(f"{modified_args}")
    g_full_expr = '\n\t'.join(g_expr)
    v_full_set = ', '.join(f_vars)
    v_full_set_1 = ', '.join(f_vars_1)
    op_call_1= target_name.split('::')[1]
    nn_op_call_1,op_call_2 = get_nn_module(str(op_call_1))
    g_full_expr_1 = op_call_2 + '%' + "\"(" + ', '.join(g_expr_1) + ")\"" + ":\"[" + ', '.join(g_expr) + "]\""
    if g_full_expr_1 not in node_ops_global_var_1:
        node_ops_global_var_1.append(g_full_expr_1)
        target = g_expr_3
        filegen_count += 1
        if yaml_gen:
            torch.save(target, f"repro_codes_1/file_{filegen_count}.pt")
            g_full_expr_1 = op_call_2 + '%' + "\"(" + ', '.join(g_expr_1) + ")\"" + ": \"[" + "file_" + str(filegen_count) + ".pt]\""
            node_ops_global_var.append(g_full_expr_1)
    else:
        return
    out_filename = f"repro_codes_1/graph_{lazy_handle_id}_{node_name}.py"
    fd = open(out_filename, 'w')
    code = f"""
import numpy
import torch
import torch.nn as nn
from torch_sendnn import torch_sendnn

def get_op_callable():
\treturn {nn_op_call_1}

class OpModule(nn.Module):
\tdef __init__(self):
\t\tsuper().__init__()
\t\top=get_op_callable()
\t\tself.op = op
\tdef forward(self, *args):
\t\treturn self.op(*args)

def SimpleIterate(device=None):
\tnet = OpModule()
\tbackend = "inductor"
\tif device == "sendnn":
\t\tbackend = 'sendnn'
\tnet_compile = torch.compile(net, backend=backend)
\t{g_full_expr}
\tinputs = [{v_full_set}]
\toutput = net_compile(*inputs)
\tif isinstance(output, tuple):
\t\toutput = output[0]
\treturn output.detach()

out_cpu = SimpleIterate()
out_sen = SimpleIterate(device="sendnn")

atol=1e-5
rtol=1e-4
same = torch.allclose(out_cpu, out_sen, atol=atol, rtol=rtol)
print("Output match:", same)

if not same:
\tprint("Max diff:", (out_cpu - out_sen).abs().max().item())
    """
    fd.write(code)
    fd.close()
    print(f"Generated all op reproduction test code for {node_name} at: {out_filename}")

def create_minimal_reproductions(lazy_handle_id, lazy_handle, all_ops, yaml_gen):
    for node in lazy_handle.graph.nodes:
        #IS_DYNAMIC = False
        if node.name not in all_ops:
            continue

        target_name = node.target.name() if hasattr(node.target, 'name') else node.target.__name__
        args = [sanitize_arg(i) for i in node._args]
        generate_reproduction(lazy_handle_id, node.name, target_name, args, yaml_gen)

def generate_yaml_file():
    node_ops_global_var.sort()
    out_filename = f"repro_codes_1/additional_shapes.yaml"
    # Example backward operators
    backward_ops = [
        "nn.functional.gelu",
        "tanh",
        "nn.functional.layer_norm",
        "nn.functional.linear",
        "log_softmax",
        "softmax",
    ]
    custom_functions = [
        "nn.functional.embedding: TestfuncEmbedding",
        "nn.functional.layer_norm: TestfuncLayernorm",
        "nn.functional.linear: TestfuncLinear",
        "log_softmax: TestfuncLogSoftmax",
        "softmax: TestfuncSoftmax",
    ]
    code = "backward_operators:\n"

    # Add backward operators
    for op in backward_ops:
        code += f"  - {op}\n"
    code += "\ncustom_functions:\n"
    for func in custom_functions:
        code += f"  - {func}\n"

    code += "\nadditional_shapes:\n"

    seen_ops = {}
    for i in node_ops_global_var:
        op, rest = i.split("%")
        if op not in seen_ops:
            code += f"  - {op}:\n"
            code += f"      - dtype: torch.bfloat16\n"
            code += f"      - params:\n"
            code += f"          - {rest}\n"
            seen_ops[op] = True
        else:
            code += f"          - {rest}\n"

    # Write to file
    with open("repro_codes_1/additional_shapes.yaml", "w") as fd:
        fd.write(code)

def generate_repro_code_all_ops(lazy_handles, yaml_gen):
    os.makedirs("repro_codes_1", exist_ok=True)
    for iter_idx, lh in enumerate(lazy_handles):
        all_ops = get_all_ops(lh)
        repro_code_generated = create_minimal_reproductions(iter_idx, lh, all_ops, yaml_gen)
    if yaml_gen:
        generate_yaml_file()
def run_allopstestgen_mode(
        model_path, model_type, yaml_gen
):
    """Runs the unsupported ops mode using the flags specified by the user."""
    setup_model_handler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
        safe_warmup=True,
    )
    preserved_lazy_handles = __state.lazy_handles
    generate_repro_code_all_ops(preserved_lazy_handles, yaml_gen)
