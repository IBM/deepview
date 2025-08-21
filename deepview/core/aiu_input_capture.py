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


def run_model_for_inputs(model_type, model_path, layer_inputs_file):
    """Generates a minimal Python script to capture per layer inputs in the
      layer io divergence mode for AIU run of FMS models.

    The generated code loads the model, compiles it using the `sendnn` backend,
    and runs inference twice to simulate lazy compilation and execution. It uses hooks to capture the inputs into a dict.

    Args:
        model_type (str): Model type (FMS/HF)
        modelpath (str): Path to the model checkpoint.
        layer_inputs_file (str): Name of pkl file with extension in which inputs are to be stored.

    Returns:
        str: A complete Python script as a string that can be executed to generate the inputs.
    """
    return f"""
from deepview.utils.ModelHandler.model_handler_utils import setup_model_handler
import torch_sendnn
import pickle
import torch
import os

os.environ["COMPILATION_MODE"] = "offline_decoder"

aiu_model_handler = setup_model_handler(
                        model_type="{model_type}",
                        model_path="{model_path}",
                        device="aiu",
                        prompt="What is the capital of Egypt?",
                        safe_warmup=False,
                        insert_forward_hooks=True
                    )

print("Reached second infer call post compile.....")
aiu_model_handler.clear_layer_io()
aiu_model_handler.infer()

print(f"Saving layer inputs.....")
aiu_model_handler.get_layer_io()
layer_inputs = aiu_model_handler.layer_inputs
with open("{layer_inputs_file}", "wb") as f:
    pickle.dump(layer_inputs, f)
print("Saved inputs to {layer_inputs_file}")

aiu_model_handler.remove_forward_hooks()
aiu_model_handler.clear_layer_io()
"""
