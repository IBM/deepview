import os
import torch

from torch_sendnn.backends import lazy_handles

node_ops_global_var=[]

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
            return f"torch.rand({arg.meta['val'].shape},requires_grad=True)"
        elif "int" in str(arg.meta["val"].dtype):
            return f"torch.randint(1, {arg.meta['val'].shape})"
        elif "bool" in str(arg.meta["val"].dtype):
            return f"torch.rand({arg.meta['val'].shape}) < 0.9"
        else:
            print(f"Unhandled arg.dtype {arg.meta['val'].dtype}")
            return str(arg)

    return str(arg)

NN_NAME_MAP = {
    "relu": "ReLU",
    "gelu": "GELU",
    "softmax": "Softmax",
    "layernorm": "LayerNorm",
    "groupnorm": "GroupNorm",
    "embedding": "Embedding"
    # add more as needed
}

def get_nn_module(op: str):
    op_name = op.lower()
    class_name = NN_NAME_MAP.get(op_name, op_name.capitalize())
    if op_name == "_unsafe_view":
        return "lambda x, shape: torch.ops.aten._unsafe_view(x, shape)"
    elif op_name == "expand":
        return "lambda x, *sizes: x.expand(*sizes)"
    elif op_name == "where.self":
        return "lambda cond, x, y: torch.where(cond, x, y)"
    elif op_name == "transpose.int":
        return "lambda x, dim0, dim1: torch.transpose(x, dim0, dim1)"
    elif op_name == "squeeze.dim":
        return "lambda x, dim: torch.squeeze(x, dim)"
    elif op_name == "slice.tensor":
        return "lambda x, dim, start, end: torch.narrow(x, dim, start, min(end, x.size(dim)) - start)"
    elif op_name == "add.scalar":
        return "lambda x, s: torch.add(x, s)"
    elif op_name == "add.tensor":
        return "lambda x, y: torch.add(x, y)"
    elif op_name == "sub.scalar":
        return "lambda x, s: torch.sub(x, s)"
    elif op_name == "sub.tensor":
        return "lambda x, y: torch.sub(x, y)"
    elif op_name == "mul.scalar":
        return "lambda x, s: torch.mul(x, s)"
    elif op_name == "mul.tensor":
        return "lambda x, y: torch.mul(x, y)"
    elif op_name == "view":
        return "lambda x, *shape: x.view(*shape)"
    elif op_name == "eq.scalar":
        return "lambda x, s: torch.eq(x, s)"
    elif op_name == "eq.tensor":
        return "lambda x, y: torch.eq(x, y)"
    elif op_name == "embedding":
        return "lambda weight, indices: torch.nn.functional.embedding(indices, weight)"
    elif op_name == "nll_loss_forward":
        return "torch.ops.aten.nll_loss_forward"
    elif op_name == "nll_loss_backward":
        return "torch.ops.aten.nll_loss_forward"
    elif hasattr(torch.nn,class_name):
        if class_name == "Embedding":
            return "getattr(torch.nn, \"" + class_name + "\")(num_embeddings=1000, embedding_dim=64)"
        else:
            return "getattr(torch.nn, \"" + class_name + "\")()"
    elif hasattr(torch, op_name):
        return "getattr(torch, \""+ op_name + "\")"
    else:
        #raise ValueError(f"Unsupported op: {op_name}")
        print(f"Unsupported op: {op_name}")
        return "unsupported_op"

def generate_reproduction(lazy_handle_id, node_name, target_name, args):
    f_vars = []
    g_expr = []
    f_vars_1 = []
    for i in range(len(args)):
        f_vars.append(f"var{i}")
        f_vars_1.append(f"var{i}.cpu()")
        f_vars_1.append(f"var{i}.grad")
        g_expr.append(f"var{i} = {args[i]}")
    g_full_expr = '\n\t'.join(g_expr)
    v_full_set = ', '.join(f_vars)
    v_full_set_1 = ', '.join(f_vars_1)
    op_call_1= target_name.split('::')[1]
    g_full_expr_1 = ', '.join(g_expr) + ', ' + op_call_1
    if g_full_expr_1 not in node_ops_global_var:
        node_ops_global_var.append(g_full_expr_1)
    else:
        return
    nn_op_call_1 = get_nn_module(str(op_call_1))
    out_filename = f"repro_codes/graph_{lazy_handle_id}_{node_name}.py"
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

def prepare_inputs(inputs):
\tprocessed_inputs = []
\trequires_grad_flags = []
\tfor inp in inputs:
\t\tif isinstance(inp, torch.Tensor):
\t\t\tinp = inp.detach().clone()
\t\t\tif inp.dtype.is_floating_point or inp.dtype.is_complex:
\t\t\t\tinp.requires_grad_()
\t\t\t\trequires_grad_flags.append(True)
\t\t\telse:
\t\t\t\trequires_grad_flags.append(False)
\t\t\tprocessed_inputs.append(inp)
\t\telse:
\t\t\tprocessed_inputs.append(inp)
\t\t\trequires_grad_flags.append(False)
\treturn processed_inputs, requires_grad_flags

def SimpleIterate(device=None):
\tnet = OpModule()
\tbackend = "inductor"
\tif device == "sendnn":
\t\tbackend = 'sendnn'
\tnet_compile = torch.compile(net, backend=backend)
\t{g_full_expr}
\tnet_compile.train()
\tinputs = [{v_full_set}]
\tinputs_clone, req_grad_flags = prepare_inputs(inputs)
\toutput = net_compile(*inputs_clone)
\tgrads = None
\tif isinstance(output, torch.Tensor) and output.requires_grad:
\t\toutput.sum().backward()
\t\tgrads = [
\t\t\tx.grad.clone() if isinstance(x, torch.Tensor) and req_grad and x.grad is not None else None
\t\t\tfor x, req_grad in zip(inputs_clone, req_grad_flags)
\t\t]
\tif isinstance(output, tuple):
\t\toutput = output[0]
\treturn output.detach(),grads

out_cpu,grad_cpu = SimpleIterate()
out_sen,grad_sen = SimpleIterate(device="sendnn")

atol=1e-5
rtol=1e-4
if grad_cpu and grad_sen:
\tfor i, (g1, g2) in enumerate(zip(grad_cpu, grad_sen)):
\t\tif g1 is not None and g2 is not None:
\t\t\tsame_grad = torch.allclose(g1, g2, atol=atol, rtol=rtol)
\t\t\tprint(f"  [GRAD MATCH] input {{i}}: {{same_grad}} | max diff = {{(g1 - g2).abs().max().item()}}")
    """
    fd.write(code)
    fd.close()
    print(f"Generated unsupported op reproduction test code for {node_name} at: {out_filename}")

def create_minimal_reproductions(lazy_handle_id, lazy_handle, all_ops):
    for node in lazy_handle.graph.nodes:
        #IS_DYNAMIC = False
        if node.name not in all_ops:
            continue

        target_name = node.target.name() if hasattr(node.target, 'name') else node.target.__name__
        args = [sanitize_arg(i) for i in node._args]
        generate_reproduction(lazy_handle_id, node.name, target_name, args)

def generate_repro_code_all_ops():
    os.makedirs("repro_codes", exist_ok=True)
    for iter_idx, lh in enumerate(lazy_handles):
        all_ops = get_all_ops(lh)
        repro_code_generated = create_minimal_reproductions(iter_idx, lh, all_ops)

