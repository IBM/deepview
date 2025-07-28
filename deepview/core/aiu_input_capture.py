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
# Standard
import os
import pickle
import subprocess

# Third Party
import torch


def generate_layerwise_inputs_aiu(
    model_type, model_path, deepview_mode, layer_inputs_file
):
    layer_inputs = None
    model_run = run_model_for_inputs(
        model_type, model_path, deepview_mode, layer_inputs_file
    )
    command1 = ["python3", "-c", model_run]
    process = subprocess.run(
        command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="")
    print(
        "DEEPVIEW========================================================================\n"
        f"DEEPVIEW Input capture for {model_path} ran succesfully.\n"
        "DEEPVIEW========================================================================\n"
    )
    if process.returncode != 0:
        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW \033[1mError running {model_path}\n\033[0m\n"
            "DEEPVIEW========================================================================\n"
        )
    with open(layer_inputs_file, "rb") as f:
        layer_inputs = pickle.load(f)
    return layer_inputs


def run_model_for_inputs(model_type, model_path, deepview_mode, layer_inputs_file):
    """Generates a minimal Python script to generate per layer outputs on precaptured inputs in the
      layer io divergence mode.

    The generated code loads the model, compiles the specified sub-layer using the `sendnn` backend,
    and runs inference twice to simulate lazy compilation and execution.

    Args:
        modelpath (str): Path to the model checkpoint.
        sub_layer (str): The sub-layer (module) name to compile and test.
        filename (str): Name of pkl file without extension in which inputs to the sublayer are stored.

    Returns:
        str: A complete Python script as a string that can be saved and executed to reproduce the failure.
    """
    return f"""
from deepview.utils.model_handler import ModelHandler
import torch_sendnn
import pickle
import torch
import os

os.environ["COMPILATION_MODE"] = "offline_decoder"

aiu_model_handler = ModelHandler(
                        model_type="{model_type}",
                        model_path="{model_path}",
                        device="aiu",
                        prompt="What is the capital of Egypt?",
                    )
aiu_model_handler.load_and_compile_model()
aiu_model_handler.prep_input()
aiu_model_handler.insert_forward_hooks("{deepview_mode}")
aiu_model_handler.warmup()

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
