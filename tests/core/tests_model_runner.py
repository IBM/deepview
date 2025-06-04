import os
import sys
import pytest
from deepview.core.model_runner import set_environment, process_output_unsupported_ops, PrintOutput

@pytest.fixture
def model_output_file(tmp_path):
  """
  Fixture to provide a temporary model_output file path.
  This will be used to test the deletion of the model_output file.
  """
  model_output_file = tmp_path / "model_output.txt"
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  tee_stdout = PrintOutput(model_output_file, original_stdout)
  tee_stderr = PrintOutput(model_output_file, original_stderr)
  sys.stdout = tee_stdout
  sys.stderr = tee_stderr
  print(model_output_file)
  return model_output_file

def test_set_environment(model_output_file):
  set_environment()
  assert os.path.exists(model_output_file) == True
  assert os.environ['DTLOG_LEVEL'] == 'error'
  assert os.environ['TORCH_SENDNN_LOG'] == 'CRITICAL'
  assert os.environ['DT_DEEPRT_VERBOSE'] == '-1'
  assert os.environ['PYTHONUNBUFFERED'] == '1'

def test_process_output_unsupported_ops(debugger_path, model_output_file, monkeypatch):
  monkeypatch.chdir(debugger_path.parent)
  print(f"Debugger path ==> {debugger_path.read_text(encoding='utf-8')}")
  print(f"Model Ouput File ==> {model_output_file.read_text(encoding='utf-8')}")
  process_output_unsupported_ops(debugger_path.read_text(encoding='utf-8'), model_output_file, True)
  assert "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n".strip() in debugger_path.read_text(encoding='utf-8')

def test_process_output_unsupported_ops_with_nodes(debugger_path, model_output_file, monkeypatch):
  monkeypatch.chdir(debugger_path.parent)
  process_output_unsupported_ops(debugger_path.read_text(encoding='utf-8'), model_output_file, True)
  assert "DEBUG TOOL Unsupported operations list:\n" in debugger_path.read_text(encoding='utf-8')
  assert all(node in tool_output_file.read_text(encoding='utf-8') for node in unique_unknown_nodes)
