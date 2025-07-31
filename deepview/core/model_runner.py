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
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import pickle
import sys

# Third Party
from torch_sendnn import backends
import torch

# Local
from deepview.core.layer_debugging import run_individual_layers
from deepview.core.layer_io_debugging import (
    SUCCESS,
    generate_layerwise_inputs_aiu,
    generate_layerwise_output_diffs,
    get_layer_thresholds,
    get_layerwise_outputs_cpu,
    get_thresholds_json_file,
)
from deepview.core.unsupported_ops import process_unsupported_ops
from deepview.utils.logger import save_deepview_logs
from deepview.utils.model_handler import ModelHandler, validate_model_id
from deepview.utils.tee import Tee


def set_environment():
    """Sets environment variables for consistent logging and output behavior during model execution.

    Also removes any pre-existing model output file (`model_output.txt`) to ensure a clean run.
    """
    old_output_file = "model_output.txt"
    if os.path.exists(old_output_file):
        os.remove(old_output_file)
    os.environ["DTLOG_LEVEL"] = "error"
    os.environ["TORCH_SENDNN_LOG"] = "CRITICAL"
    os.environ["DT_DEEPRT_VERBOSE"] = "-1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["COMPILATION_MODE"] = "offline_decoder"


def run_unsupported_op_mode(
    model_path, model_type, show_details_flag, generate_repro_code_flag
):
    """Runs the unsupported ops mode using the flags specified by the user."""
    aiu_model_handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
    )
    aiu_model_handler.load_and_compile_model()
    aiu_model_handler.prep_input()
    aiu_model_handler.safe_warmup()
    process_unsupported_ops(show_details_flag, generate_repro_code_flag)


def run_layer_debugging_mode(model_path, model_type, generate_repro_code_flag):
    """Runs the layer debugging mode using the flags specified by the user."""
    aiu_model_handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
    )
    aiu_model_handler.load_and_compile_model()
    aiu_model_handler.prep_input()
    aiu_model_handler.insert_forward_hooks()
    aiu_model_handler.safe_warmup()

    print(f"Saving layer inputs.....")
    aiu_model_handler.get_layer_io()
    layer_inputs = aiu_model_handler.layer_inputs
    inputs_filename = model_path.split("/")[-1] + ".pkl"
    with open(f"{inputs_filename}", "wb") as f:
        pickle.dump(layer_inputs, f)
    print(f"Saved inputs to {inputs_filename}")

    aiu_model_handler.remove_forward_hooks()
    aiu_model_handler.clear_layer_io()

    run_individual_layers(aiu_model_handler, inputs_filename, generate_repro_code_flag)


def run_layer_io_divergence_mode(model_path, model_type):
    """Runs the layer_io_divergence_mode mode. Uses inputs_filename to get the precaptured inputs if specified by the user.

    Returns True if all layers pass the thresholds test, otherwise False.
    """
    theshold_filepath = get_thresholds_json_file(model_path)
    if not theshold_filepath:
        print(f"Unable to find thresholds for {model_path}.")
        sys.exit(0)
    print("Getting layer output thresholds.....")
    thesholds = get_layer_thresholds(theshold_filepath)

    print("========= Running on CPU to capture layer IO ==========")
    cpu_model_handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device="cpu",
        prompt="What is the capital of Egypt?",
    )
    cpu_model_handler.load_and_compile_model()
    cpu_model_handler.prep_input()
    cpu_model_handler.insert_forward_hooks()
    print("Reached first infer call post compile.....")
    cpu_model_handler.infer()
    print(f"Getting layerwise outputs.....")
    cpu_model_handler.get_layer_io()
    cpu_layer_outputs = get_layerwise_outputs_cpu(cpu_model_handler)
    cpu_model_handler.remove_forward_hooks()
    cpu_model_handler.clear_layer_io()

    print(cpu_layer_outputs.keys())

    print("========= Running on AIU to capture layer divergence ==========")
    aiu_model_handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
    )
    print("Capturing layerwise inputs....")
    inputs_filename = model_path.split("/")[-1] + ".pkl"
    aiu_model_handler.layer_inputs = generate_layerwise_inputs_aiu(
        model_type, model_path, inputs_filename
    )
    if not aiu_model_handler.layer_inputs:
        print(f"Input capture failed for {model_path}.")
        sys.exit(0)
    print("Capturing layerwise outputs and calculating divergence....")
    diverging_layer, status = generate_layerwise_output_diffs(
        aiu_model_handler, inputs_filename, cpu_layer_outputs, thesholds
    )
    if diverging_layer is None and status == SUCCESS:
        print(
            f"DEEPVIEW Threshold test passed for all layers of {model_path}\n"
            "DEEPVIEW========================================================================\n"
        )
        return True
    return False


def run_model(
    model_type,
    model_path,
    tool_output_file,
    deepview_mode,
    show_details_flag,
    generate_repro_code_flag,
    logfile="model_output.txt",
):
    """Main entry point to run a model and process its execution logs.

    Loads and compiles the model using `ModelHandler`, runs inference.

    For `layer_debugging` mode, it inserts hooks and runs individual layers.
    For `unsupported_op` mode, it detects unsupported ops in the model.
    For `aiu_input_capture` mode, it inserts hooks and captures inputs of each layer.
    For `layer_io_divergence` mode, it runs the model on cpu and captures the layerwise.
    output divergence by running each layer one-by-one on AIU using precaptured inputs.

    Args:
        model_type (str): Type of the model - hf or fms.
        model_path (str): Path to the model checkpoint.
        tool_output_file (str): Path to store filtered and analyzed output (e.g., unsupported ops or failed layers).
        deepview_mode (str): Mode to run DeepView in; can include 'layer_debugging' or 'unsupported_op'.
        show_details_flag (str): Whether to print the stack trace for unsupported ops.
        generate_repro_code_flag (bool): Whether to generate reproducible test scripts for failure points.
        logfile (str, optional): File to store full execution logs. Defaults to 'model_output.txt'.
    """

    # First check that the model_path is valid.
    if not validate_model_id(model_path):
        raise ValueError(f"Invalid model ID or FMS path: {model_path}")

    # Prints generated by compile_and_infer are shown in terminal as well as saved in logfile.
    with open(logfile, "a") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee), redirect_stderr(tee):
            if generate_repro_code_flag:
                backends.preserve_lazyhandle()

            torch.set_default_dtype(torch.float16)

            if deepview_mode == "unsupported_op":
                run_unsupported_op_mode(
                    model_path, model_type, show_details_flag, generate_repro_code_flag
                )

            elif deepview_mode == "layer_debugging":
                run_layer_debugging_mode(
                    model_path,
                    model_type,
                    generate_repro_code_flag,
                )

            elif deepview_mode == "layer_io_divergence":
                passed = run_layer_io_divergence_mode(model_path, model_type)

            # Process the logfile to create the tool_output_file
            tee.flush()
            save_deepview_logs(logfile, tool_output_file)
