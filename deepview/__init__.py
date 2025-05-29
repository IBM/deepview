"""
DeepView: A tool for debugging ML models
"""

import os
from importlib.metadata import version, PackageNotFoundError

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

# Import key components for easier access
from deepview.core.model_runner import run_model, set_environment
from deepview.core.hook_monitor import enable_unsupported_op_mode, clear_unsupported_op_mode