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

# Standard
import os

# Third Party
import pytest

# Local
from deepview.core.hook_monitor import (
    clear_unsupported_op_mode,
    enable_unsupported_op_mode,
)


@pytest.fixture
def show_details_flag():
    """
    A fixture to control the show_details flag is passed on the deepview CLI for testing.
    returns True if enabled in the CLI with `--show_details`, but can be overridden in tests if needed.

    Returns:
        bool: True to mock `--show_details` is enabled.
    """
    return True


def test_enable_unsupported_op_mode(show_details_flag):
    """
    Test to ensure the environment variables for unsupported operation mode are set correctly.

    Args:
        show_details_flag: A fixture that controls whether the `--show_details` is enabled.
    """
    enable_unsupported_op_mode(show_details_flag)
    assert os.environ.get("UNSUP_OP") == "1"
    assert os.environ.get("UNSUP_OP_DEBUG") == ("1" if show_details_flag else "0")


def test_clear_unsupported_op_mode():
    """
    Test to ensure the environment variables for unsupported operation mode are cleared correctly.
    This test checks that the environment variables are reset to their default values.
    """
    clear_unsupported_op_mode()
    assert os.environ.get("UNSUP_OP") == "0"
    assert os.environ.get("UNSUP_OP_DEBUG") == "0"
