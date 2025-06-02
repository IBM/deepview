# Standard
import os
import re
import shutil

# Third Party
from sendnn import opcodes
from torch_sendnn.backends import lazy_handles
import torch


def get_unsupported_ops(lazy_handle):
    unsupported_ops = []
    g2 = lazy_handle.g2 if hasattr(lazy_handle, "g2") else lazy_handle.meta["g2"]
    for op in g2.compute_ops:
        if op.Fn() == opcodes.Unsupported:
            unsupported_ops.append(op.Name())
    return unsupported_ops


def sanitize_arg(arg):
    # pdb.set_trace()
    if isinstance(arg, list):
        return "[" + ", ".join([sanitize_arg(i) for i in arg]) + "]"

    if isinstance(arg, torch.fx.node.Node):
        if "float" in str(arg.dtype):
            return f"torch.rand({arg.shape})"
        elif "int" in str(arg.dtype):
            return f"torch.randint(1, {arg.shape})"
        elif "bool" in str(arg.dtype):
            return f"torch.rand({arg.shape}) < 0.9"
        else:
            # raise Exception("Unhandled arg.dtype {arg.dtype}")
            print(f"Unhandled arg.dtype {arg.dtype}")
            return str(arg)

    return str(arg)


def generate_reproduction(lazy_handle_id, node_name, target_name, args):
    f_vars = []
    g_expr = []
    for i in range(len(args)):
        f_vars.append(f"var{i}")
        g_expr.append(f"var{i} = {args[i]}")
    g_full_expr = "\n\t".join(g_expr)
    v_full_set = ", ".join(f_vars)
    op_call = "torch.ops." + target_name.replace("::", ".") + "(" + v_full_set + ")"

    code = f"""
import torch
from torch_sendnn import torch_sendnn

def f({v_full_set}):
\treturn {op_call}

def g():
\t{g_full_expr}
\tf_compile = torch.compile(f, backend="sendnn")
\treturn f_compile({v_full_set})

g()    
    """

    out_filename = f"repro_codes/graph_{lazy_handle_id}_{node_name}.py"

    with open(out_filename, "w") as fd:
        fd.write(code)

    print(
        f"Generated unsupported op reproduction test code for {node_name} at: {out_filename}"
    )


def create_minimal_reproductions(lazy_handle_id, lazy_handle, unsupported_ops):
    if hasattr(lazy_handle, "ori_gm"):
        ori_gm = lazy_handle.ori_gm
    else:
        ori_gm = lazy_handle.meta["original_gm"]

    for node in ori_gm.graph.nodes:
        if node.name not in unsupported_ops:
            continue
        target_name = (
            node.target.name() if hasattr(node.target, "name") else node.target.__name__
        )
        args = [sanitize_arg(i) for i in node._args]
        generate_reproduction(lazy_handle_id, node.name, target_name, args)


def generate_repro_code_unsupported_ops():
    os.makedirs("repro_codes", exist_ok=True)
    for iter_idx, lh in enumerate(lazy_handles):
        unsupported_ops = get_unsupported_ops(lh)
        repro_code_generated = create_minimal_reproductions(
            iter_idx, lh, unsupported_ops
        )


def generate_repro_code_layer_debugging(err_msg, layer, modelpath):
    match = re.search(r"input shape (\[[^\]]+\]), data type (\S+)", err_msg)
    input_str = match.group(1)
    dtype_str = match.group(2)

    src_template, dst_repro = (
        "core/template_repro_code.py",
        f"{layer.split('.')[-1]}_repro_code.py",
    )

    try:
        shutil.copy(src_template, dst_repro)
        with open(dst_repro, "r") as f:
            content = (
                f.read()
                .replace("modelpath", "'" + modelpath + "'")
                .replace("sub_layer", layer)
                .replace("input_shape", input_str)
                .replace("datatype", dtype_str)
            )
        with open(dst_repro, "w") as f:
            f.write(content)
        print(f"The repro code is stored in file {dst_repro}")
    except FileNotFoundError:
        print(f"Error: Template file not found: {src_template}")
