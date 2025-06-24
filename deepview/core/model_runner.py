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
import re
import sys
import pickle
from datetime import datetime

# Third Party
from torch_sendnn import torch_sendnn
import torch

# Local
from deepview.core.input_output_debugging import generate_individual_layer_output
from deepview.core.layer_debugging import run_individual_layers
from deepview.core.unsupported_ops import process_unsupported_ops
from deepview.utils.logger import save_deepview_logs
from deepview.utils.model_handler import ModelHandler
from deepview.utils.tee import Tee


def set_environment():
    """Sets environment variables for consistent logging and output behavior during model execution.

    Also removes any pre-existing model output file (`model_output.txt`) to ensure a clean run.
    """
    old_output_files = ["model_output.txt", "input_kwargs.pth", "output_kwargs.pth"]
    for file in old_output_files:
        if os.path.exists(file):
            print(f"Deleting the old {file}..............")
            os.remove(file)
    os.environ["DTLOG_LEVEL"] = "error"
    os.environ["TORCH_SENDNN_LOG"] = "CRITICAL"
    os.environ["DT_DEEPRT_VERBOSE"] = "-1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["COMPILATION_MODE"] = "offline_decoder"


def run_unsupported_op_mode(aiu_model_handler, show_details_flag, generate_repro_code_flag):
    aiu_model_handler.safe_warmup()
    process_unsupported_ops(show_details_flag, generate_repro_code_flag)


def run_layer_debugging_mode(aiu_model_handler,deepview_mode, model_path, model_type, generate_repro_code_flag):
    aiu_model_handler.insert_forward_hooks(deepview_mode)
    aiu_model_handler.safe_warmup()
    aiu_model_handler.remove_forward_hooks()
    with open("model_list.txt", "w") as file:
        json.dump(
            {k: list(v) for k, v in aiu_model_handler.layer_list.items()},
            file,
        )
    run_individual_layers(
        model_path,
        model_type,
        aiu_model_handler.layer_list,
        generate_repro_code_flag,
    )


def run_io_dumping_mode(aiu_model_handler,deepview_mode, model_path, model_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ## AIU run
    aiu_model_handler.insert_forward_hooks(deepview_mode)
    aiu_model_handler.warmup()
    print("Reached second infer call post compile.....")
    aiu_model_handler.infer()
    print("Reached third infer call post compile.....")
    aiu_model_handler.infer()
    aiu_model_handler.get_layer_io()
    aiu_model_handler.remove_forward_hooks()
    aiu_layer_io = generate_individual_layer_output(
        aiu_model_handler,
        model_path,
        model_type,
        'aiu',
        timestamp
    )

    ## CPU run
    print("========= AIU IO captured. Running on CPU ==========")
    cpu_model_handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device='cpu',
        prompt="What is the capital of India?",
    )
    cpu_model_handler.load_and_compile_model()
    cpu_model_handler.prep_input()
    cpu_model_handler.insert_forward_hooks(deepview_mode)
    cpu_model_handler.infer()
    cpu_model_handler.get_layer_io()
    cpu_model_handler.remove_forward_hooks()
    cpu_layer_io = generate_individual_layer_output(
        cpu_model_handler,
        model_path,
        model_type,
        'cpu',
        timestamp
    )

    ## TODO: Flavia to add code here. aiu_layer_io and cpu_layer_io are the lists of dictionaries used to store layer name, inputs and outputs
    # from AIU and CPU runs, respectively.


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

    Args:
        model_type (str): Type of the model - hf or fms.
        model_path (str): Path to the model checkpoint.
        tool_output_file (str): Path to store filtered and analyzed output (e.g., unsupported ops or failed layers).
        deepview_mode (str): Mode to run DeepView in; can include 'layer_debugging' or 'unsupported_op'.
        show_details_flag (str): Whether to print the stack trace for unsupported ops.
        generate_repro_code_flag (bool): Whether to generate reproducible test scripts for failure points.
        logfile (str, optional): File to store full execution logs. Defaults to 'model_output.txt'.
    """
    # Prints generated by compile_and_infer are shown in terminal as well as saved in logfile.
    with open(logfile, "a") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee), redirect_stderr(tee):
            if generate_repro_code_flag:
                torch_sendnn.preserve_lazyhandle()

            torch.set_default_dtype(torch.float16)

            aiu_model_handler = ModelHandler(
                model_type=model_type,
                model_path=model_path,
                device='aiu',
                prompt="What is the capital of India?",
            )
            aiu_model_handler.load_and_compile_model()
            aiu_model_handler.prep_input()

            print("Reached first infer call post compile.....")
            try:
                if deepview_mode == "unsupported_op":
                    run_unsupported_op_mode(aiu_model_handler, show_details_flag, generate_repro_code_flag)
                    
                if deepview_mode == "layer_debugging":
                    run_layer_debugging_mode(aiu_model_handler,deepview_mode, model_path, model_type, generate_repro_code_flag)

                if deepview_mode == "io_dump":
                    run_io_dumping_mode(aiu_model_handler,deepview_mode, model_path, model_type)
            except Exception as e:
                print(f"Exception occurred: {e}")

            # Process the logfile to create the tool_output_file
            tee.flush()
            save_deepview_logs(logfile, tool_output_file)
