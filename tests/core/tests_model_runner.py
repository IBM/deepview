import os
import pytest
from deepview.core.model_runner import set_environment, process_output_unsupported_ops


@pytest.fixture
def model_output_file(tmp_path):
    """
    Fixture to provide a temporary model_output file path.
    This will be used to test the deletion of the model_output file.

    Args:
        tmp_path: A pytest fixture that provides a temporary directory for testing.
    """
    model_output_file = tmp_path / "model_output.txt"
    print(model_output_file)
    return model_output_file


def test_set_environment(model_output_file):
    """
    Test to ensure the environment variables are set correctly.
    This test checks that the old model output file is deleted if it exists,
    and that the environment variables are set to the expected values.

    Args:
        model_output_file: A fixture that provides a temporary file path for the model output.
    """
    set_environment()
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
    process_output_unsupported_ops(
        debugger_path.read_text(encoding="utf-8"), model_output_file, True
    )
    assert (
        "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n".strip()
        in debugger_path.read_text(encoding="utf-8")
    )


def test_process_output_unsupported_ops_with_nodes(
    debugger_path, model_output_file, monkeypatch
):
    """
    Test to ensure the unsupported operations are correctly processed from the debugger output when there are unsupported ops.
    This test checks if the `process_output_unsupported_ops` function correctly identifies unsupported operations
    and generates the expected output in the debugger file, including a list of unique unsupported nodes.

    Args:
        debugger_path: A fixture that runs the model and generates the output file.
        model_output_file: A fixture that provides a temporary file path for the model output.
        monkeypatch: Used to change the current working directory to the debugger output file's parent directory and not clutter the env with files generated.
    """
    monkeypatch.chdir(debugger_path.parent)
    process_output_unsupported_ops(
        debugger_path.read_text(encoding="utf-8"), model_output_file, True
    )
    assert "DEBUG TOOL Unsupported operations list:\n" in debugger_path.read_text(
        encoding="utf-8"
    )
    assert all(
        node in tool_output_file.read_text(encoding="utf-8")
        for node in unique_unknown_nodes
    )
