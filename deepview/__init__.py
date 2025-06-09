"""
DeepView: A tool for debugging ML models
"""

# Standard
from importlib.metadata import PackageNotFoundError, version
import os

# Get version from VERSION.txt
try:
    version_file = os.path.join(os.path.dirname(__file__), "VERSION.txt")
    with open(version_file, "r") as f:
        __version__ = f.read().strip()
except FileNotFoundError:
    try:
        __version__ = version("deepview")
    except PackageNotFoundError:
        __version__ = "unknown"

# Local
from deepview.core.hook_monitor import (
    clear_unsupported_op_mode,
    enable_unsupported_op_mode,
)

# Import key components for easier access
from deepview.core.model_runner import run_model, set_environment
