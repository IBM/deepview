# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

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
