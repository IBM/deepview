# Injects hooks at runtime to catch unsupported ops, shapes, and types
# Standard
import os


def enable_unsupported_op_mode(show_details_flag):
    """Enables runtime mode to print unsupported operations in torch_sendnn layer.

    Sets environment variables to activate unsupported op detection.
    Optionally enables detailed debug output if `show_details_flag` is True.

    Args:
        show_details_flag (bool): If True, enables verbose debug output for unsupported ops.
    """
    os.environ["UNSUP_OP"] = "1"
    os.environ["UNSUP_OP_DEBUG"] = "0"
    if show_details_flag:
        os.environ["UNSUP_OP_DEBUG"] = "1"


def clear_unsupported_op_mode():
    """Disables the unsupported operation detection mode.

    Resets environment variables to turn off unsupported op tracking and debug output.
    """
    os.environ["UNSUP_OP"] = "0"
    os.environ["UNSUP_OP_DEBUG"] = "0"
