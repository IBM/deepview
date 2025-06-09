# Standard
import os
import re
import shutil

# Third Party
from sendnn import opcodes
from torch_sendnn.backends import lazy_handles
import torch


def get_unsupported_ops(lazy_handle):
    """Retrieves a list of operations marked as unsupported from the lazy handle's g2 graph.

    Args:
        lazy_handle: An object containing the sengraph (`g2`).

    Returns:
        list[str]: Names of operations in this lazy handle's g2 graph that are marked as unsupported.
    """
    unsupported_ops = []
    g2 = lazy_handle.g2 if hasattr(lazy_handle, "g2") else lazy_handle.meta["g2"]
    for op in g2.compute_ops:
        if op.Fn() == opcodes.Unsupported:
            unsupported_ops.append(op.Name())
    return unsupported_ops


def sanitize_arg(arg):
    """Converts an argument into a string representation that can be used in auto-generated repro code.

    Args:
        arg (Any): The argument to sanitize. Can be a list, a `torch.fx.Node`, or a primitive value.

    Returns:
        str: A Python-valid string representing the argument, approximated for test code generation.
    """
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
    """Creates a minimal Python script to reproduce the behavior of a specific unsupported operation.

    Args:
        lazy_handle_id (int): Index or identifier of the lazy handle for file naming.
        node_name (str): Name of the node corresponding to the unsupported operation.
        target_name (str): Fully qualified name of the target PyTorch op (e.g., "aten::add").
        args (list[str]): Sanitized string representations of the input arguments.
    """
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
    """Generates minimal test scripts for each unsupported ops found in the lazy handle using `generate_reproduction`.

    Args:
        lazy_handle_id (int): Index of the lazy handle, used for output file naming.
        lazy_handle: Object containing the original graph module (`ori_gm`) or its metadata.
        unsupported_ops (list[str]): List of operation node names marked as unsupported.
    """
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
    """Generates repro scripts for all unsupported operations found in the lazy handles.

    It first retrieves the list of unsupported operations from each lazy handle
    using the `get_unsupported_ops()` function, and then generates minimal repro
    scripts for each using the `create_minimal_reproductions()` function.
    """
    os.makedirs("repro_codes", exist_ok=True)
    for iter_idx, lh in enumerate(lazy_handles):
        unsupported_ops = get_unsupported_ops(lh)
        repro_code_generated = create_minimal_reproductions(
            iter_idx, lh, unsupported_ops
        )
