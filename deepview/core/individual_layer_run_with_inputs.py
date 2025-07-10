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
import torch

def run_layers_with_inputs(modelpath, sub_layer, filename):
    """Generates a minimal Python script to reproduce a layer-level failure in layer debugging mode.

    The generated code loads the model, compiles the specified sub-layer using the `sendnn` backend,
    and runs inference twice to simulate lazy compilation and execution.

    Args:
        modelpath (str): Path to the model checkpoint.
        sub_layer (str): The sub-layer (module) name to compile and test.
        input_shape (str): Input tensor shape.
        datatype (str): Torch datatype for the input tensor (e.g., 'torch.float16').

    Returns:
        str: A complete Python script as a string that can be saved and executed to reproduce the failure.
    """
    return f"""
from fms.models import get_model
from torch import tensor
import torch_sendnn
import inspect
import pickle
import torch
import os

os.environ["COMPILATION_MODE"] = "offline_decoder"
torch.compiler.reset()
torch._dynamo.reset()
torch._dynamo.reset_code_caches()

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

layer = {sub_layer}
target_layer = layer
forward_signature = inspect.signature(target_layer.forward)
expected_args = list(forward_signature.parameters.keys())

input_filename = "temp/{filename}_input.pkl"
with open(input_filename, 'rb') as f:
    inputval = pickle.load(f) 
inputvals = list(inputval)
if len(inputval) < len(expected_args):
    zipped_inputs = list(itertools.zip_longest(expected_args, inputval, fillvalue=None))
else:
    zipped_inputs = list(zip(expected_args, inputval))

kwargs = dict(zipped_inputs)

layer.compile(backend="sendnn", dynamic=False)
with torch_sendnn.warmup_mode():
    result = layer(**kwargs)
    
result = layer(**kwargs)
    
input_kwargs_filename = "temp/{filename}_input_kwargs.pkl"
output_filename = "temp/{filename}_output_kwargs.pkl"

with open(input_kwargs_filename, 'wb') as f:
    pickle.dump(kwargs, f) 
with open(output_filename, 'wb') as f:
    pickle.dump(result, f) 
"""