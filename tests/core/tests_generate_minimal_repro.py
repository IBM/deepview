import pytest
from core.generate_minimal_repro import get_unsupported_ops

# Dummy classes to simulate behavior of ops and lazy_handle
class DummyOp:
  def __init__(self, name, fn_return):
    self._name = name
    self._fn_return = fn_return

  @property
  def Name(self):
    return self._name

  def Fn(self):
    return self._fn_return

class DummyG2:
  def __init__(self, compute_ops):
    self.compute_ops = compute_ops

class DummyLazyHandle:
  def __init__(self, g2):
    self.g2 = g2

@pytest.fixture
def lazy_handle():
  op1 = DummyOp("op1", "Unsupported")
  op2 = DummyOp("op2", "Supported")
  g2 = DummyG2([op1, op2])
  return DummyLazyHandle(g2)

@pytest.mark.test_get_unsupported_ops
def test_get_unsupported_ops(lazy_handle):
  result = get_unsupported_ops(lazy_handle)
  assert result == ["op1"]
