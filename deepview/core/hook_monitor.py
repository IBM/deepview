# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

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
