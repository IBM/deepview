# Standard
from pathlib import Path
import subprocess


def run_individual_layers(model_path, model_type, layer_list, generate_repro_code_flag):
    """Runs each unique layer of the model individually in layer_debugging mode.

    Iterates over the provided layers and attempts to compile and run each one in isolation.
    Stops at the first failure and optionally generates a minimal reproducible script for debugging.

    Args:
        model_path (str): Path to the model checkpoint directory.
        model_type (str): Model type, either 'hf' (HuggingFace) or 'fms' (Foundation Model Stack).
        layer_list (dict): Dictionary mapping layer/module names to a set containing input shape and data type.
        generate_repro_code_flag (bool): If True, generates a minimal repro script when a layer fails.
    """
    print("Running each layer individually........")
    layers_done = []
    failed_layer = "No failed layer"
    for str_layer, val in layer_list.items():
        sub_layer = (
            str_layer.rsplit(".", str_layer.count(".") - 3)[0]
            if str_layer.count(".") > 3
            else str_layer
        )
        val_list = list(val)
        datatype, input_shape = val_list if "torch" in val_list[0] else val_list[::-1]
        if sub_layer in layers_done:
            continue

        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW Running {sub_layer}, {input_shape}, {datatype}"
        )

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

        layer_run = run_layers(model_path, sub_layer, input_shape, datatype)
        command1 = ["python3", "-c", layer_run]
        process = subprocess.run(
            command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        for line in process.stdout:
            print(line, end="")
        if process.returncode != 0:
            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW \033[1mError running {sub_layer}, {input_shape}, {datatype}\n\033[0m"
                "DEEPVIEW========================================================================\n"
            )
            failed_layer = sub_layer
            break
        else:
            print(
                f"DEEPVIEW Successfully ran {sub_layer}, {input_shape}, {datatype}\n"
                "DEEPVIEW========================================================================\n"
            )

        layers_done.append(sub_layer)

    if failed_layer != "No failed layer":
        if generate_repro_code_flag:
            generate_repro_code_layer_debugging(
                model_path, model_type, failed_layer, input_shape, datatype
            )


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
