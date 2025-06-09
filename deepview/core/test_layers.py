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


def run_layers(model_path, model_type):
    """Returns a Python script that loads the model and runs inference on each layer individually.

    It reads `model_list.txt` for layers’ input shapes and data types, then runs each unique layer
    once, printing debug information.

    Args:
        model_path (str): Path to the model checkpoint.
        model_type (str): Type of the model - hf or fms.

    Returns:
        str: Python script for layer-wise execution.
    """
    return f"""
# Standard
import argparse
import json
import os
import sys

# Third Party
from fms.models import get_model
from torch_sendnn import torch_sendnn
import torch

# Local
from deepview.utils.model_handler import ModelHandler

try:
    with open("model_list.txt", "r") as file:
        layer_list = json.load(file)
except FileNotFoundError:
    print("File 'model_list.txt' not found")
    sys.exit(1)
except json.JSONDecodeError:
    print("Invalid JSON format")
    sys.exit(1)

model_handler = ModelHandler(
    model_type='{model_type}',
    model_path='{model_path}',
    prompt="What is the capital of India?",
)
model = model_handler.load_and_compile_model()

layers_done = []
# Run inference for each layer
for str_layer, val in layer_list.items():
    sub_layer = (
        str_layer.rsplit(".", str_layer.count(".") - 3)[0]
        if str_layer.count(".") > 3
        else str_layer
    )

    if sub_layer in layers_done:
        continue

    # Determine input shape and data type
    dtype_str, input_shape_str = val if "torch" in val[0] else val[::-1]

    input_shape = eval(input_shape_str)
    data_type = eval(dtype_str)
    rand_tensor = torch.rand(tuple(input_shape))

    # Compile and run
    layer = eval(sub_layer)
    layer.compile(backend="sendnn_decoder", dynamic=False)

    print(
        "-------------------------------------------------------------------------------------------"
    )
    print(
        f"DEBUG TOOL first run for {{sub_layer}}, input shape {{input_shape_str}}, data type {{dtype_str}}"
    )
    layer(rand_tensor.to(data_type))

    print(
        f"DEBUG TOOL update lazy handle for {{sub_layer}}, input shape {{input_shape_str}}, data type {{dtype_str}}"
    )
    torch_sendnn.update_lazyhandle()

    print(
        f"DEBUG TOOL second run for {{sub_layer}}, input shape {{input_shape_str}}, data type {{dtype_str}}"
    )
    layer(rand_tensor.to(data_type))
    print(
        "-------------------------------------------------------------------------------------------"
    )

    layers_done.append(sub_layer)
"""
