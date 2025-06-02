# Injects hooks at runtime to catch unsupported ops, shapes, and types
# Standard
import os


def enable_unsupported_op_mode(show_details_flag):
    os.environ["UNSUP_OP"] = "1"
    os.environ["UNSUP_OP_DEBUG"] = "0"
    if show_details_flag:
        os.environ["UNSUP_OP_DEBUG"] = "1"


def clear_unsupported_op_mode():
    os.environ["UNSUP_OP"] = "0"
    os.environ["UNSUP_OP_DEBUG"] = "0"
