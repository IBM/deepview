# Standard
from pathlib import Path
import subprocess

# Local
from deepview.core.individual_layer_run import run_layers


def run_individual_layers(logfile, model_path, model_type, layer_list):
    """Runs each layer of the model individually (in layer debugging mode).

    Args:
        logfile (str): Path to the complete model output log.
        model_path (str): Path to the model checkpoint.
        model_type (str): Type of model hf or fms.
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
        # Show output in terminal as well as save in file
        with open(logfile, "a") as f:
            start_line = (
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW Running {sub_layer}, {input_shape}, {datatype}"
            )
            print(start_line)
            layer_run = run_layers(model_path, sub_layer, input_shape, datatype)
            command1 = ["python3", "-c", layer_run]
            process = subprocess.run(
                command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")
            if process.returncode != 0:
                error_line = (
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW Error running {sub_layer}, {input_shape}, {datatype}\n"
                    "DEEPVIEW========================================================================\n"
                )
                failed_layer = sub_layer
                print(error_line)
                break
            else:
                success_line = (
                    f"DEEPVIEW Successfully ran {sub_layer}, {input_shape}, {datatype}\n"
                    "DEEPVIEW========================================================================\n"
                )
                print(success_line)
        layers_done.append(sub_layer)
    return failed_layer, input_shape, datatype


def process_output_layer_debugging(
    tool_output_file,
    logfile,
    generate_repro_code_flag,
    model_path,
    failed_layer,
    input_str,
    dtype_str,
):
    """Parses the model execution log to identify which layer failed in layer debugging mode.

    If a failure is detected in the model's forward pass at a specific layer, the layer name and
    failure message are printed and stored. Optionally triggers reproduction code generation.

    Args:
        tool_output_file (str): Output file to store DEEPVIEW lines and failure summary.
        logfile (str): Path to the complete model output log.
        generate_repro_code_flag (bool): Whether to generate reproduction code for the failing layer.
        model_path (str): Path to the model checkpoint to use in generating the repro code.
    """
    # All DEEPVIEW output lines are extracted and saved in tool_output_file.

    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        debug_lines = [line for line in infile if line.startswith("DEEPVIEW")]
        outfile.writelines(debug_lines)
        # Trigger repro code generation
        if failed_layer != "No failed layer":
            if generate_repro_code_flag:
                generate_repro_code_layer_debugging(
                    model_path, failed_layer, input_str, dtype_str
                )


def generate_repro_code_layer_debugging(modelpath, layer, input_str, dtype_str):
    """Generates layer-specific repro code from an error message for layer_debugging mode.

    Args:
        err_msg (str): Error string containing input shape and data type information.
        layer (str): The layer name (dotted path) in the model where the error occurred.
        modelpath (str): Path to the model checkpoint.
    """
    dst_repro = f"{layer.split('.')[-1]}_repro_code.py"
    try:
        Path(dst_repro).touch()
        with open(dst_repro, "w") as f:
            f.write(run_layers(modelpath, layer, input_str, dtype_str))
        print(f"The repro code is stored in file {dst_repro}\n")
    except Exception as e:
        print(f"Error: Repro code generation : {e}")
