import pytest
import argparse
from deepview.core.model_runner import run_model

@pytest.fixture(scope="session", autouse=True)
def debugger_path(tmp_path_factory):
  args = argparse.Namespace(
    model_type='fms',
    model_path='/mnt/aiu-models-en-shared/models/hf/Mistral-7B-Instruct-v0.3',
    deepview_mode=['unsupported_op'],
    show_details=False,
    generate_repro_code_flag=False,
    tool_output_file=tmp_path_factory.getbasetemp() / 'test_debugger.txt'
  )

  run_model(args.model_type, args.model_path, args.tool_output_file, args.deepview_mode, args.generate_repro_code_flag)
  debugger_path = tmp_path_factory.getbasetemp() / "test_debugger.txt"
  return debugger_path