import os
import torch

from torch_sendnn.backends import lazy_handles

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
            return f"{arg.meta['val'].shape}$torch.rand({arg.meta['val'].shape},requires_grad=True)"
        elif "int" in str(arg.meta["val"].dtype):
            return f"{arg.meta['val'].shape}$torch.randint(1, {arg.meta['val'].shape})"
        elif "bool" in str(arg.meta["val"].dtype):
            return f"{arg.meta['val'].shape}$torch.rand({arg.meta['val'].shape}) < 0.9"
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

def get_nn_module(op: str, test_gen):
    op_name = op.lower()
    class_name = NN_NAME_MAP.get(op_name, op_name.capitalize())
    if op_name == "_unsafe_view":
        return "lambda x, shape: torch.ops.aten._unsafe_view(x, shape)", "_unsafe"
    elif op_name == "expand":
        return "lambda x, *sizes: x.expand(*sizes)", "expand"
    elif op_name == "where.self":
        return "lambda cond, x, y: torch.where(cond, x, y)", "where"
    elif op_name == "transpose.int":
        return "lambda x, dim0, dim1: torch.transpose(x, dim0, dim1)", "transpose"
    elif op_name == "squeeze.dim":
        return "lambda x, dim: torch.squeeze(x, dim)", "squeeze"
    elif op_name == "slice.tensor":
        return "lambda x, dim, start, end: torch.narrow(x, dim, start, min(end, x.size(dim)) - start)", "slice"
    elif op_name == "add.scalar":
        return "lambda x, s: torch.add(x, s)", "add"
    elif op_name == "add.tensor":
        return "lambda x, y: torch.add(x, y)", "add"
    elif op_name == "sub.scalar":
        return "lambda x, s: torch.sub(x, s)", "sub"
    elif op_name == "sub.tensor":
        return "lambda x, y: torch.sub(x, y)", "sub"
    elif op_name == "mul.scalar":
        return "lambda x, s: torch.mul(x, s)", "mul"
    elif op_name == "mul.tensor":
        return "lambda x, y: torch.mul(x, y)", "mul"
    elif op_name == "view":
        return "lambda x, *shape: x.view(*shape)", "view"
    elif op_name == "eq.scalar":
        return "lambda x, s: torch.eq(x, s)", "eq"
    elif op_name == "eq.tensor":
        return "lambda x, y: torch.eq(x, y)", "eq"
    elif op_name == "embedding":
        return "lambda weight, indices: torch.nn.functional.embedding(indices, weight)", "nn.functional.embedding"
    elif op_name == "nll_loss_forward":
        return "torch.ops.aten.nll_loss_forward", "nll_loss_forward"
    elif op_name == "nll_loss_backward":
        return "torch.ops.aten.nll_loss_backward", "nll_loss_backward"
    elif op_name == "native_layer_norm_backward":
        return "torch.ops.aten.native_layer_norm_backward", "native_layer_norm_backward"
    elif op_name == "gelu_backward":
        return "torch.ops.aten.gelu_backward", "gelu_backward"
    elif op_name == "tanh_backward":
        return "torch.ops.aten.tanh_backward", "tanh_backward"
    elif op_name == "select_backward":
        return "torch.ops.aten.select_backward", "select_backward"
    elif op_name == "slice_backward":
        return "torch.ops.aten.slice_backward", "slice_backward"
    elif op_name == "sum.dim_intlist":
        return "lambda x, dim, keepdim=False: torch.sum(x, dim=dim, keepdim=keepdim)", "sum"
    elif hasattr(torch.nn,class_name):
        if class_name == "Embedding":
            return "getattr(torch.nn, \"" + class_name + "\")(num_embeddings=1000, embedding_dim=64)", "nn.functional.embedding"
        else:
            return "getattr(torch.nn, \"" + class_name + "\")()", "nn.functional." + op_name
    elif hasattr(torch, op_name):
        return "getattr(torch, \""+ op_name + "\")", op_name
    else:
        #raise ValueError(f"Unsupported op: {op_name}")
        if test_gen:
            print(f"Invalid test case generated for this: {op_name}. Need to add support for this op in get_nn_module function")
        return "unsupported_op", op_name

def generate_reproduction(lazy_handle_id, node_name, target_name, args, test_gen, yaml_gen):
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
    nn_op_call_1,op_call_2 = get_nn_module(str(op_call_1), test_gen)
    g_full_expr_1 = op_call_2 + '%' + "\"(" + ', '.join(g_expr_1) + ")\"" + ":\"[" + ', '.join(g_expr) + "]\""
    if g_full_expr_1 not in node_ops_global_var_1:
        node_ops_global_var_1.append(g_full_expr_1)
        target = g_expr_3
        filegen_count += 1
        if yaml_gen:
            torch.save(target, f"repro_codes/file_{filegen_count}.pt")
            g_full_expr_1 = op_call_2 + '%' + "\"(" + ', '.join(g_expr_1) + ")\"" + ": \"[" + "file_" + str(filegen_count) + ".pt]\""
            node_ops_global_var.append(g_full_expr_1)
    else:
        return
    if not test_gen:
        return
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
    print(f"Generated all op reproduction test code for {node_name} at: {out_filename}")

def create_minimal_reproductions(lazy_handle_id, lazy_handle, all_ops, test_gen,yaml_gen):
    for node in lazy_handle.graph.nodes:
        #IS_DYNAMIC = False
        if node.name not in all_ops:
            continue

        target_name = node.target.name() if hasattr(node.target, 'name') else node.target.__name__
        args = [sanitize_arg(i) for i in node._args]
        generate_reproduction(lazy_handle_id, node.name, target_name, args, test_gen, yaml_gen)

def generate_yaml_file():
    node_ops_global_var.sort()
    out_filename = f"repro_codes/additional_shapes.yaml"
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
    with open("repro_codes/additional_shapes.yaml", "w") as fd:
        fd.write(code)

def generate_repro_code_all_ops(test_gen,yaml_gen):
    os.makedirs("repro_codes", exist_ok=True)
    for iter_idx, lh in enumerate(lazy_handles):
        all_ops = get_all_ops(lh)
        repro_code_generated = create_minimal_reproductions(iter_idx, lh, all_ops, test_gen, yaml_gen)
    if yaml_gen:
        generate_yaml_file()

