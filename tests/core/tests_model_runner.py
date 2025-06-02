import os
import pytest
from deepview.core.model_runner import set_environment, process_output_unsupported_ops

@pytest.fixture
def model_output_file(tmp_path):
  """
  Fixture to provide a temporary model_output file path.
  This will be used to test the deletion of the model_output file.
  """
  return tmp_path / 'model_output.txt'
  model_output_file = tmp_path / 'model_output.txt'
  return model_output_file

def test_set_environment(model_output_file):
  set_environment()
  assert os.path.exists(model_output_file) == False
  assert os.environ['DTLOG_LEVEL'] == 'error'
  assert os.environ['TORCH_SENDNN_LOG'] == 'CRITICAL'
  assert os.environ['DT_DEEPRT_VERBOSE'] == '-1'
  assert os.environ['PYTHONUNBUFFERED'] == '1'

def test_process_output_unsupported_ops(debugger_path):
  process_output_unsupported_ops(debugger_path, model_output_file, True)
  if len(unique_unknown_nodes) == 0:
    assert "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n" in tool_output_file.read_text(encoding='utf-8')
  elif len(unique_unknown_nodes) > 0:
    assert "DEBUG TOOL Unsupported operations list:\n" in tool_output_file.read_text(encoding='utf-8')
    assert all(node in tool_output_file.read_text(encoding='utf-8') for node in unique_unknown_nodes)
