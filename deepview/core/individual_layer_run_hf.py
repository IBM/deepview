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


def run_layers(modelpath, sub_layer, filename):
    """Generates a minimal Python script to reproduce a layer-level failure in layer debugging mode for HF models.

    The generated code loads the model, compiles the specified sub-layer using the `sendnn` backend,
    and runs inference twice to simulate lazy compilation and execution.

    Args:
        modelpath (str): Path to the model checkpoint.
        sub_layer (str): The sub-layer (module) name to compile and test.
        filename (str): Name of pkl file with extension in which inputs to the sublayer are stored.

    Returns:
        str: A complete Python script as a string that can be saved and executed to reproduce the failure.
    """
    return f"""
from deepview.utils.ModelHandler.model_handler_utils import create_model_handler
from torch import tensor
import torch_sendnn
import itertools
import pickle
import inspect
import torch
import os

os.environ["COMPILATION_MODE"] = "offline"

model_handler = create_model_handler(
    model_type='hf',
    model_path='{modelpath}',
    device="aiu",
    prompt='What is the capital of Egypt?',
)

model_handler.load_model()

model = model_handler.model
model.eval()
torch.set_grad_enabled(False)

{sub_layer}.compile(backend="sendnn", dynamic=False)

forward_signature = inspect.signature({sub_layer}.forward)
expected_args = list(forward_signature.parameters.keys())


with open("{filename}", "rb") as f:
    layer_ios_dict = pickle.load(f)
inputval = layer_ios_dict["{sub_layer}"]["input"]
inputvals = list(inputval)
kwargs = layer_ios_dict["{sub_layer}"]["kwarg"]

if 'attn_kwargs' in expected_args:
    expected_args.remove('attn_kwargs')

all_keys = list(expected_args) + [k for k in kwargs if k not in expected_args]

all_kwargs = {{
    k: inputvals[i] if i < len(inputvals) else kwargs.get(k)
    for i, k in enumerate(all_keys)
}} 

with torch_sendnn.warmup_mode():
    result = {sub_layer}(**all_kwargs) 

print(f"Warmup for {sub_layer} completed")
result = {sub_layer}(**all_kwargs)
print(f"Second run for {sub_layer} completed")
"""
