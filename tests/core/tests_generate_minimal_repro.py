import pytest
from sendnn import opcodes
from deepview.core.generate_minimal_repro import get_unsupported_ops

@pytest.mark.test_no_unsupported_ops_found
def test_no_unsupported_ops_found(lazy_handle):
  result = get_unsupported_ops(lazy_handle)
  assert result == []

@pytest.mark.test_unsupported_ops_found
def test_unsupported_ops_found(lazy_handle):
  result = get_unsupported_ops(lazy_handle)
  assert len(result) > 0
  assert all(isinstance(op, str) for op in result)

@pytest.mark.test_ops_found_are_unsupported
def test_ops_found_are_unsupported(lazy_handle):
  result = get_unsupported_ops(lazy_handle)
  assert all(op.Fn() == opcodes.Unsupported for op in result)
