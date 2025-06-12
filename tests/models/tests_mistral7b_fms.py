import pytest
import argparse
from deepview.core.model_runner import run_model


@pytest.fixture(scope="session", autouse=True)
def debugger_path(tmp_path_factory):
    """
    A fixture to run the model and generate the debugger output file.
    This fixture is automatically used in tests that require the debugger output.
    It runs the model with the specified arguments and saves the output to a temporary file.
    It same as call `deepview --model_type <model_type> --model <model_path> --mode <deepview_mode> --show_details --generate_repro_code --output_file <tool_output_file>`
    """
    args = argparse.Namespace(
        model_type="fms",
        model_path="/mnt/aiu-models-en-shared/models/mistralai/Mistral-7B-Instruct-v0.3",
        deepview_mode=["unsupported_op", "layer_debugging"],
        show_details=True,
        generate_repro_code_flag=True,
        tool_output_file=tmp_path_factory.getbasetemp() / "test_debugger.txt",
    )

    run_model(
        args.model_type,
        args.model_path,
        args.tool_output_file,
        args.deepview_mode,
        args.generate_repro_code_flag,
    )
    debugger_path = tmp_path_factory.getbasetemp() / "test_debugger.txt"
    return debugger_path


def test_debugger_output_exits(debugger_path):
    """
    Test to ensure the debugger output file exists after running the model.
    debugger_path is a fixture that runs the model and generates the output file.
    """
    assert (
        debugger_path.exists()
    ), f"Debugger output file {debugger_path} does not exist."


def test_get_unsupported_ops_with_no_ops_found(debugger_path):
    """
    Test to ensure the unsupported operations are correctly extracted from the debugger output.
    This test checks if the `get_unsupported_ops` function returns a empty list of unsupported operations.
    """
    assert (
        "No unsupported operations detected." in debugger_path.read_text()
    ), "Expected 'No unsupported operations detected.' in debugger output."
