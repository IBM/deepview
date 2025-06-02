import pytest
import os
from deepview.core.hook_monitor import enable_unsupported_op_mode, clear_unsupported_op_mode

@pytest.fixture
def show_details_flag():
  """
  A fixture to control the show_details flag is passed on the deepview CLI for testing.
  returns True if enabled in the CLI with `--show_details`, but can be overridden in tests if needed.
  """
  return True

def test_enable_unsupported_op_mode(show_details_flag):
  enable_unsupported_op_mode(show_details_flag)
  assert os.environ.get('UNSUP_OP') == '1'
  assert os.environ.get('UNSUP_OP_DEBUG') == ('1' if show_details_flag else '0')


def test_clear_unsupported_op_mode():
  clear_unsupported_op_mode()
  assert os.environ.get('UNSUP_OP') == '0'
  assert os.environ.get('UNSUP_OP_DEBUG') == '0'
