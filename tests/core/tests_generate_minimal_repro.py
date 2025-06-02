import pytest
from deepview.core.generate_minimal_repro import generate_repro_code_unsupported_ops

@pytest.mark.test_generate_repro_code_unsupported_ops
def test_generate_repro_code_unsupported_ops(debugger_path):
  assert debugger_path.exists()

@pytest.mark.test_unsupported_ops_found
def test_unsupported_ops_found(debugger_path):
  output = debugger_path.read_text(encoding='utf-8')
  assert "Operation not supported" in output

@pytest.mark.test_unsupported_ops_listed
def test_unsupported_ops_listed(debugger_path):
  output = debugger_path.read_text(encoding='utf-8')
  assert "Unsupported operations list" in output
