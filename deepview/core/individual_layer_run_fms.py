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
# Third Party
import torch


def run_layers(modelpath, sub_layer, filename):
    """Generates a minimal Python script to reproduce a layer-level failure in layer debugging mode for FMS models.

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
from fms.models import get_model
from torch import tensor
import itertools
import torch_sendnn
import inspect
import pickle
import torch
import os

os.environ["COMPILATION_MODE"] = "offline"

model = get_model(
    "hf_pretrained",
    variant='{modelpath}',
    device_type="cpu",
    data_type=torch.float16,
    source=None,
    distributed_strategy=None,
    linear_config={{"linear_type": "torch_linear"}},
    fused_weights=False,
)

device = torch.device("cpu")
model.eval()
torch.set_grad_enabled(False)

{sub_layer}.compile(backend="sendnn", dynamic=False)

with open("{filename}", "rb") as f:
    layers_ios_dict = pickle.load(f)
args = layers_ios_dict["{sub_layer}"]["args"]
kwargs = layers_ios_dict["{sub_layer}"]["kwargs"]

forward_signature = inspect.signature({sub_layer}.forward)
expected_args = list(forward_signature.parameters.keys())

if 'attn_kwargs' in expected_args:
    if 'mask' in kwargs:
        kwargs['attn_kwargs'] = kwargs['mask']

### The following two lines are required for running this mode on Llama model. 
# TODO: Check if there is a way to remove the need for this in FMS 
if 'reverse' in kwargs.keys():
    kwargs['reverse'] = True

with torch_sendnn.warmup_mode():
    result = {sub_layer}(*args, **kwargs) 
print(f"Warmup for {sub_layer} completed")
result = {sub_layer}(*args, **kwargs) 
print(f"Second run for {sub_layer} completed")
"""
