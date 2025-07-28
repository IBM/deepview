# Standard
from pathlib import Path
import os
import pickle
import re
import subprocess

# Local
from deepview.core.individual_layer_run_fms import run_layers


def convert_attr_path(attr_path):
    """Converts the name of the modules to match the format in thresholds file."""
    attr_path = "model." + attr_path

    def replace_numeric_attr(match):
        number = match.group(1)
        tail = match.group(2)
        return f"[{number}]{tail}"

    pattern = re.compile(r"\.(\d+)(\.|$)")
    converted = pattern.sub(replace_numeric_attr, attr_path)
    return converted


def run_individual_layers(aiu_model_handler, inputs_filename, generate_repro_code_flag):
    """Runs each unique layer of the model individually in layer_debugging mode.

    Iterates over the provided layers and attempts to compile and run each one in isolation.
    Stops at the first failure and optionally generates a minimal reproducible script for debugging.

    Args:

        generate_repro_code_flag (bool): If True, generates a minimal repro script when a layer fails.
    """
    model = aiu_model_handler.model
    layers_done = []
    failed_layer = "No failed layer"

    print("Running each layer individually........")
    for str_layer in aiu_model_handler.layer_inputs.keys():
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = "model"
        if sub_layer in layers_done:
            continue
        # if sub_layer != "model" and sub_layer != "model.base_model":
        # if sub_layer != "model.base_model":
        layer_run = run_layers(
            aiu_model_handler.model_path, sub_layer, str_layer, inputs_filename
        )

        command1 = ["python3", "-c", layer_run]
        process = subprocess.run(
            command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW Layer is {sub_layer}."
        )
        if process.returncode != 0:
            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW \033[1mError running {sub_layer}\n\033[0m"
                "DEEPVIEW========================================================================\n"
            )
            failed_layer = sub_layer
            break
        else:
            print(
                f"DEEPVIEW Successfully ran {sub_layer}\n"
                "DEEPVIEW========================================================================\n"
            )

        layers_done.append(sub_layer)

    # if failed_layer != "No failed layer":
    #     if generate_repro_code_flag:
    #         generate_repro_code_layer_debugging(
    #             model_path, model_type, failed_layer, input_shape, datatype
    #         )


def generate_repro_code_layer_debugging(
    model_path, model_type, layer, input_str, dtype_str
):
    """Generates and saves layer-level repro script for debugging failures in layer_debugging mode.

    This function creates a standalone Python script that reproduces the issue for the specified layer
    by compiling and running it with given input shape and data type.

    Args:
        model_path (str): Path to the model checkpoint.
        model_type (str): Model type, either 'hf' (HuggingFace) or 'fms' (Foundation Model Stack).
        layer (str): Python expression referring to the layer where the failure occurred.
        input_str (str): String representation of the input tensor shape (e.g., '(1, 64, 64)').
        dtype_str (str): PyTorch data type as a string (e.g., 'torch.float32').
    """
    if model_type == "fms":
        # Local
        from deepview.core.individual_layer_run_fms import run_layers
    elif model_type == "hf":
        # Local
        from deepview.core.individual_layer_run_hf import run_layers
    else:
        print(
            "DEEPVIEW \033[1mError running individual layers - only fms and hf models area supported\n\033[0m"
        )
        return

    dst_repro = f"{layer.split('.')[-1]}_repro_code.py"
    try:
        Path(dst_repro).touch()
        with open(dst_repro, "w") as f:
            f.write(run_layers(model_path, layer, input_str, dtype_str))
        print(f"The repro code is stored in file {dst_repro}\n")
    except Exception as e:
        print(f"Error: Repro code generation : {e}")
