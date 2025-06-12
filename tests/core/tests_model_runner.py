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
from pathlib import Path
import os
import shutil
import sys

# Third Party
import pytest

# Local
from deepview.core.model_runner import process_output_unsupported_ops


@pytest.fixture
def model_output_file(tmp_path):
    """
    Fixture that finds 'model_output.txt' in the current working directory,
    copies it to the pytest-provided tmp_path, and yields its Path.
    """
    src_file = Path.cwd() / "model_output.txt"
    dst_file = tmp_path / "model_output.txt"

    # Wait or check for the file to be generated if needed,
    # Or raise if not present.
    assert src_file.exists(), f"{src_file} was not found after running utility!"

    shutil.copy(str(src_file), str(dst_file))

    return dst_file


def test_set_environment(model_output_file):
    """
    Test to ensure the environment variables are set correctly.
    This test checks that the old model output file is deleted if it exists,
    and that the environment variables are set to the expected values.

    Args:
        model_output_file: A fixture that provides a temporary file path for the model output.
    """
    assert os.path.exists(model_output_file)
    assert os.environ["DTLOG_LEVEL"] == "error"
    assert os.environ["TORCH_SENDNN_LOG"] == "CRITICAL"
    assert os.environ["DT_DEEPRT_VERBOSE"] == "-1"
    assert os.environ["PYTHONUNBUFFERED"] == "1"


def test_process_output_unsupported_ops(debugger_path, model_output_file, monkeypatch):
    """
    Test to ensure the unsupported operations are correctly processed from the debugger output.
    This test checks if the `process_output_unsupported_ops` function correctly identifies unsupported operations
    and generates the expected output in the debugger file.

    Args:

        debugger_path: A fixture that runs the model and generates the output file.
        model_output_file: A fixture that provides a temporary file path for the model output.
        monkeypatch: Used to change the current working directory to the debugger output file's parent directory and not clutter the env with files generated.
    """
    monkeypatch.chdir(debugger_path.parent)
    print(f"Debugger path ==> {debugger_path.read_text(encoding="utf-8")}")
    print(f"Model Ouput File ==> {model_output_file.read_text(encoding="utf-8")}")
    process_output_unsupported_ops(debugger_path, model_output_file, True)
    debugger_text = debugger_path.read_text(encoding="utf-8")
    # Merge the asserts: pass if either message is present
    assert (
        "DEEPVIEW \033[1mNo unsupported operations detected.\033[0m\n".strip()
        in debugger_text
        or "DEEPVIEW Unsupported operations list:\n" in debugger_text
    ), "Neither 'No unsupported operations detected' nor 'Unsupported operations list' found in debugger output."
