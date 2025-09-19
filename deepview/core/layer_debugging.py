# Standard
from pathlib import Path
import os
import pickle
import re
import subprocess

# Local
from deepview.utils.model_handler import ModelHandler, setup_model_handler


def run_individual_layers(aiu_model_handler, inputs_filename, generate_repro_code_flag):
    """Runs each unique layer of the model individually in layer_debugging mode.

    Iterates over the provided layers and attempts to compile and run each one in isolation.
    Stops at the first failure and optionally generates a minimal reproducible script for debugging.

    Args:
        aiu_model_handler (obj): Model handler object.
        inputs_filename (str): Name of the file storing the individual layer inputs.
        generate_repro_code_flag (bool): If True, generates a minimal repro script when a layer fails.
    """
    model = aiu_model_handler.model
    layers_done = []
    failed_layer = "No failed layer"

    if aiu_model_handler.model_type == "fms":
        # Local
        from deepview.core.individual_layer_run_fms import run_layers
    elif aiu_model_handler.model_type == "hf":
        # Local
        from deepview.core.individual_layer_run_hf import run_layers
    else:
        print(
            "DEEPVIEW \033[1mError running individual layers - only fms and hf models area supported\n\033[0m"
        )
        return

    print("Running each layer individually........")

    for layer in aiu_model_handler.layers_ios.keys():
        if layer in layers_done or layer.endswith("]"):
            continue
        layer_run = run_layers(aiu_model_handler.model_path, layer, inputs_filename)

        command1 = ["python3", "-c", layer_run]
        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW Running layer {layer}."
        )
        process = subprocess.run(
            command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
        if process.returncode != 0:
            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW \033[1mError running {layer}\n\033[0m"
                "DEEPVIEW========================================================================\n"
            )
            failed_layer = layer
            break
        else:
            print(
                f"DEEPVIEW Successfully ran {layer}\n"
                "DEEPVIEW========================================================================\n"
            )
        layers_done.append(layer)

    if failed_layer != "No failed layer":
        if generate_repro_code_flag:
            generate_repro_code_layer_debugging(aiu_model_handler, failed_layer)


def generate_repro_code_layer_debugging(aiu_model_handler, failed_layer):
    """Generates and saves layer-level repro script for debugging failures in layer_debugging mode.

    This function creates a standalone Python script that reproduces the issue for the specified layer
    by compiling and running it with given input shape and data type.

    Args:
        aiu_model_handler (obj): Model handler object.
        failed_layer (str): Python string referring to the layer where the failure occurred.
    """
    if aiu_model_handler.model_type == "fms":
        # Local
        from deepview.core.individual_layer_run_fms import run_layers
    elif aiu_model_handler.model_type == "hf":
        # Local
        from deepview.core.individual_layer_run_hf import run_layers
    else:
        print(
            "DEEPVIEW \033[1mError running individual layers - only fms and hf models area supported\n\033[0m"
        )
        return
    dst_repro = f"{failed_layer.split('.')[-1]}_repro_code.py"
    try:
        Path(dst_repro).touch()
        with open(dst_repro, "w") as f:
            inputs_filename = aiu_model_handler.model_path.split("/")[-1] + ".pkl"
            f.write(
                run_layers(
                    aiu_model_handler.model_path,
                    failed_layer,
                    inputs_filename,
                )
            )
        print(f"The repro code is stored in file {dst_repro}\n")
    except Exception as e:
        print(f"Error: Repro code generation : {e}")


def run_layer_debugging_mode(model_path, model_type, generate_repro_code_flag):
    """Runs the layer debugging mode using the flags specified by the user."""
    aiu_model_handler = setup_model_handler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
        safe_warmup=True,
        insert_forward_hooks=True,
    )

    print(f"Saving layer inputs.....")
    aiu_model_handler.get_layer_io()
    layers_ios = aiu_model_handler.layers_ios
    inputs_filename = model_path.split("/")[-1] + ".pkl"
    with open(f"{inputs_filename}", "wb") as f:
        pickle.dump(layers_ios, f)
    print(f"Saved inputs to {inputs_filename}")

    aiu_model_handler.remove_forward_hooks()
    aiu_model_handler.clear_layer_io()

    run_individual_layers(aiu_model_handler, inputs_filename, generate_repro_code_flag)
