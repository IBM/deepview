# Standard
import os
import re
import shutil

# Third Party
from sendnn import opcodes
from torch_sendnn.backends.sendnn_backend import __state
from torch_sendnn.conversion.conversion_utils import (
    shape_to_list,
    torch_datatype_to_sendnn,
)
import torch

# Local
from deepview.utils.model_handler import setup_model_handler


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
        if "float" in str(arg.meta["val"].dtype):
            return f"torch.rand({arg.meta['val'].shape})"
        elif "int" in str(arg.meta["val"].dtype):
            return f"torch.randint(1, {arg.meta['val'].shape})"
        elif "bool" in str(arg.meta["val"].dtype):
            return f"torch.rand({arg.meta['val'].shape}) < 0.9"
        else:
            # raise Exception("Unhandled arg.dtype {arg.dtype}")
            print(f"Unhandled arg.dtype {arg.meta['val'].dtype}")
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
    os.makedirs("repro_codes", exist_ok=True)
    f_vars = []
    g_expr = []
    for i in range(len(args)):
        f_vars.append(f"var{i}")
        g_expr.append(f"var{i} = {args[i]}")
    g_full_expr = "\n\t".join(g_expr)
    v_full_set = ", ".join(f_vars)
    op_call = "torch.ops." + target_name.replace("::", ".") + "(" + v_full_set + ")"

    code = f"""
import os
os.environ["TORCH_SENDNN_LOG"] = "DEBUG"

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
        f"Generated unsupported op reproduction test code for {node_name} at: {out_filename}\n\n"
    )


def add_prefix_to_string(original_string):
    """Adds 'DEEPVIEW ' prefix to each line of the input string."""
    prefix = "DEEPVIEW "
    return "\n".join(prefix + line for line in original_string.split("\n"))


def process_unsupported_ops_lazy_handle(
    lazy_handle_id,
    lazy_handle,
    unsupported_ops,
    show_details_flag,
    generate_repro_code_flag,
):
    """Processes unsupported operations from a lazy handle and optionally generates reproduction scripts.

    For each unsupported operation in the graph module, this function prints diagnostic information,
    including data type, shape, and stack trace (if enabled), and optionally generates a minimal
    reproduction script for debugging.

    Args:
        lazy_handle_id (int): Identifier for the lazy handle, used in naming output files.
        lazy_handle: Object containing the original graph module.
        unsupported_ops (list[str]): List of node names that are identified as unsupported.
        show_details_flag (bool): Whether to print stack traces for each unsupported op.
        generate_repro_code_flag (bool): Whether to generate a minimal script to reproduce the error.
    """
    for node in lazy_handle.graph.nodes:
        if node.name not in unsupported_ops:
            continue

        # =================== Print the details of the unsupported ops =========================================
        # Note: This logic of finding data type and shape is taken from torch_sendnn
        IS_DYNAMIC = False
        if isinstance(node.meta["val"], list):
            dt = [torch_datatype_to_sendnn(t) for t in node.meta["val"]]
            shape = [shape_to_list(s, IS_DYNAMIC) for s in node.meta["val"]]
        else:
            dt = torch_datatype_to_sendnn(node.meta["val"].dtype)
            shape = shape_to_list(node.meta["val"].shape, IS_DYNAMIC)

        error = ""
        if show_details_flag:
            error = f"DEEPVIEW==================================== Stack Trace ====================================\n{add_prefix_to_string(node.stack_trace)}"
        print(
            f"DEEPVIEW Caught error for \033[1m{node}\033[0m: Operation not supported.\nDEEPVIEW Data type: {dt}, Shape: {shape}\n{error}"
        )
        # ======================================================================================================

        if generate_repro_code_flag:
            target_name = (
                node.target.name()
                if hasattr(node.target, "name")
                else node.target.__name__
            )
            args = [sanitize_arg(i) for i in node._args]
            generate_reproduction(lazy_handle_id, node.name, target_name, args)


def process_unsupported_ops(show_details_flag, generate_repro_code_flag):
    """Identifies unsupported operations and optionally generates reproduction scripts.

    This function processes all lazy handles to extract unsupported operations using
    `get_unsupported_ops()`, and calls `process_unsupported_ops_lazy_handle()` to
    optionally print detailed diagnostics and generate minimal repro scripts.
    It also prints a de-duplicated summary of unsupported operations at the end.

    Args:
        show_details_flag (bool): Whether to print detailed stack traces for each unsupported op.
        generate_repro_code_flag (bool): Whether to generate minimal repro scripts for each op.
    """
    lazy_handles = __state.lazy_handles
    all_unsupported_ops = []
    for iter_idx, lh in enumerate(lazy_handles):
        unsupported_ops = get_unsupported_ops(lh)
        all_unsupported_ops.extend(unsupported_ops)
        process_unsupported_ops_lazy_handle(
            iter_idx, lh, unsupported_ops, show_details_flag, generate_repro_code_flag
        )

    def strip(s):
        return re.sub(r"\x1b\[[0-9;]*m", "", s)

    seen = set()
    unique_unsupported_ops = [
        op
        for op in all_unsupported_ops
        if not re.match(r".*_\d+$", strip(op))
        and (strip(op) not in seen and not seen.add(strip(op)))
    ]

    if len(unique_unsupported_ops) == 0:
        print(
            "DEEPVIEW========================================================================\n"
            "DEEPVIEW \033[1mNo unsupported operations detected.\033[0m\n"
            "DEEPVIEW========================================================================\n\n\n"
        )
    else:
        unique_unsupported_ops = [
            "\033[1m" + op + "\033[0m" for op in unique_unsupported_ops
        ]
        unique_unsupported_ops_str = "\n".join(sorted(unique_unsupported_ops))
        print(
            "DEEPVIEW========================================================================\n"
            "DEEPVIEW Unsupported operations list:\n"
            f"{add_prefix_to_string(unique_unsupported_ops_str)}\n"
            "DEEPVIEW========================================================================\n\n\n"
        )


def run_unsupported_op_mode(
    model_path, model_type, show_details_flag, generate_repro_code_flag
):
    """Runs the unsupported ops mode using the flags specified by the user."""
    setup_model_handler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
        safe_warmup=True,
    )
    process_unsupported_ops(show_details_flag, generate_repro_code_flag)
    