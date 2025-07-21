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
import argparse

# Third Party
import pytest

# Local
from deepview.core.model_runner import run_model, set_environment

@pytest.fixture(scope="module", autouse=True, params=["unsupported_op", "layer_debugging"])
def debugger_setup(tmp_path_factory, request):
    """
    A fixture to run the model and generate the debugger output file.
    This fixture is automatically used in tests that require the debugger output.
    It runs the model with the specified arguments and saves the output to the working directory as model_output.txt.

    Args:
        tmp_path_factory: A pytest fixture that provides a temporary directory for testing.
        request: The pytest request object containing the parameter.

    Returns:
        tuple: (debugger_path, mode) - The path to the debugger output file and the mode used.
    """
    from pathlib import Path

    mode = request.param
    
    args = argparse.Namespace(
        model_type="fms",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        output_file=tmp_path_factory.getbasetemp() / "test_mistral7b_debugger.txt",
        mode=mode,
        show_details=True,
        generate_repro_code=True,
    )

    set_environment()

    run_model(
        args.model_type,
        args.model,
        args.output_file,
        args.mode,
        args.show_details,
        args.generate_repro_code,
    )
    debugger_path = tmp_path_factory.getbasetemp() / "test_mistral7b_debugger.txt"
    return debugger_path, mode

def test_debugger_output_exits(debugger_setup):
    """
    Test to ensure the debugger output file exists after running the model.

    Args:
        debugger_setup: A fixture that runs the model and generates the output file.
    """
    debugger_path, mode = debugger_setup
    print(f"Running test_debugger_output_exits with mode: {mode}")
    assert (
        debugger_path.exists()
    ), f"Debugger output file {debugger_path} does not exist."

def test_get_unsupported_ops(model_output_file, debugger_setup):
    """
    Test to ensure the unsupported operations are correctly extracted from the debugger output.
    This test only runs when mode is 'unsupported_op'.

    Args:
        model_output_file: A fixture that provides the debugger output file.
        debugger_setup: A fixture that provides the debugger path and mode.
    """
    debugger_path, mode = debugger_setup
    if mode != "unsupported_op":
        pytest.skip(f"Skipping test_get_unsupported_ops for mode: {mode}")
    
    print(f"Running test_get_unsupported_ops with mode: {mode}")
    assert (
        "DEEPVIEW \033[1mNo unsupported operations detected.\033[0m\n"
        in model_output_file.read_text(encoding="utf-8")
    ), "Expected 'DEEPVIEW \033[1mNo unsupported operations detected.\033[0m\n'"

def test_layer_debugging_mode(model_output_file, debugger_setup):
    """
    Test to ensure the layer debugging mode is correctly set in the debugger output.
    This test only runs when mode is 'layer_debugging'.

    Args:
        model_output_file: A fixture that provides the debugger output file.
        debugger_setup: A fixture that provides the debugger path and mode.
    """
    debugger_path, mode = debugger_setup
    if mode != "layer_debugging":
        pytest.skip(f"Skipping test_layer_debugging_mode for mode: {mode}")
    
    print(f"Running test_layer_debugging_mode with mode: {mode}")    
    assert "Running each layer individually" in model_output_file.read_text(
        encoding="utf-8"
    ), "Expected Running each layer individually"
    assert (
        "DEEPVIEW \033[1mError running model, [1, 1], torch.int64\n"
        in model_output_file.read_text(encoding="utf-8")
    ), "Error running model, [1, 1], torch.int64"
