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
from fms.models import get_model
import deepview.utils.model_handler as dvmh

import torch_sendnn
import itertools
import pickle
import inspect
import torch
import os
os.environ["COMPILATION_MODE"] = "offline"

model_handler = dvmh.ModelHandler(
    model_type='hf',
    model_path='{modelpath}',
    device="aiu",
    prompt='What is the capital of Egypt?',
)

model_handler.model_class = model_handler._infer_model_class('{modelpath}')
model_class = dvmh.MODEL_CLASSES[model_handler.model_class]

if model_handler.model_class == "causal_lm":
    model_handler.model = model_class.from_pretrained('{modelpath}')
else:
    model_handler.model = dvmh.AutoModel.from_pretrained('{modelpath}')

model = model_handler.model
model.eval()
torch.set_grad_enabled(False)

layer = {sub_layer}
target_layer = layer
forward_signature = inspect.signature(target_layer.forward)
expected_args = list(forward_signature.parameters.keys())


with open("{filename}", "rb") as f:
    layer_inputs_dict = pickle.load(f)
inputval = layer_inputs_dict["{sub_layer}"]
inputvals = list(inputval)

if len(inputval) < len(expected_args):
    print("WARNING: Missing values of input arguments padded with None.")
    zipped_inputs = list(itertools.zip_longest(expected_args, inputval, fillvalue=None))
else:
    zipped_inputs = list(zip(expected_args, inputval))
kwargs = dict(zipped_inputs)
    
layer.compile(backend="sendnn", dynamic=False)

with torch_sendnn.warmup_mode():
   result = layer(**kwargs)    
print(f"Warmup for {sub_layer} completed")
result = layer(**kwargs)
print(f"Second run for {sub_layer} completed")
"""
