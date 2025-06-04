# Standard
from contextlib import redirect_stderr, redirect_stdout
import json
import os
import re
import subprocess
import sys

# Third Party
from torch_sendnn import torch_sendnn
import torch

# Local
from deepview.core.generate_minimal_repro import (
    generate_repro_code_layer_debugging,
    generate_repro_code_unsupported_ops,
)
from deepview.core.test_layers import run_layers
from deepview.utils.model_handler import ModelHandler
from deepview.utils.tee import Tee


def set_environment():
    """Sets environment variables for consistent logging and output behavior during model execution.

    Also removes any pre-existing model output file (`model_output.txt`) to ensure a clean run.
    """    
    old_output_file = "model_output.txt"
    if os.path.exists(old_output_file):
        print("Deleting the old model_output.txt..............")
        os.remove(old_output_file)
    os.environ["DTLOG_LEVEL"] = "error"
    os.environ["TORCH_SENDNN_LOG"] = "CRITICAL"
    os.environ["DT_DEEPRT_VERBOSE"] = "-1"
    os.environ["PYTHONUNBUFFERED"] = "1"


def process_output_unsupported_ops(tool_output_file, logfile, generate_repro_code_flag):
    """Parses the model execution log to extract and report unsupported operations.

    It identifies lines starting with 'DEBUG TOOL' that indicate unsupported operations, 
    filters unique op names, and writes them to `tool_output_file`. Optionally triggers
    reproduction code generation if unsupported ops are found.

    Args:
        tool_output_file (str): Output file to store processed DEBUG TOOL lines and unsupported op summary.
        logfile (str): Path to the complete model output log.
        generate_repro_code_flag (bool): Whether to generate reproduction code for the unsupported ops.
    """    
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.
    unknown_nodes = []
    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        for line in infile:
            if line.startswith("DEBUG TOOL"):
                outfile.write(line)
                match = re.search(r"DEBUG TOOL Caught error for (.*?):", line)
                if match:
                    node = match.group(1)
                    unknown_nodes.append(node)

        def strip(s):
            return re.sub(r"\x1b\[[0-9;]*m", "", s)

        seen = set()
        unique_unknown_nodes = [
            op
            for op in unknown_nodes
            if not re.match(r".*_\d+$", strip(op))
            and (strip(op) not in seen and not seen.add(strip(op)))
        ]
        if len(unique_unknown_nodes) == 0:
            no_unsup_op = (
                "DEBUG TOOL========================================================================\n"
                "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n"
            )
            print(no_unsup_op)
            outfile.write(no_unsup_op)
        else:
            unknown_nodes_str = "\n".join(sorted(unique_unknown_nodes))
            final_line = (
                "DEBUG TOOL========================================================================\n"
                "DEBUG TOOL Unsupported operations list:\n"
                f"{unknown_nodes_str}"
            )
            print(final_line)
            outfile.write(final_line + "\n")

    # Generate reproduction code for unsupported ops (if any)
    if generate_repro_code_flag:
        if len(unique_unknown_nodes):
            print("Generating reproduction code")
            generate_repro_code_unsupported_ops()
        else:
            print("No reproduction code generated as no unsupported operations found.")


def process_output_layer_debugging(
    tool_output_file, logfile, generate_repro_code_flag, model_path
):
    """Parses the model execution log to identify which layer failed in layer debugging mode.

    If a failure is detected in the model's forward pass at a specific layer, the layer name and
    failure message are printed and stored. Optionally triggers reproduction code generation.

    Args:
        tool_output_file (str): Output file to store DEBUG TOOL lines and failure summary.
        logfile (str): Path to the complete model output log.
        generate_repro_code_flag (bool): Whether to generate reproduction code for the failing layer.
        model_path (str): Path to the model checkpoint to use in generating the repro code.
    """    
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.

    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        debug_lines = [line for line in infile if line.startswith("DEBUG TOOL")]
        outfile.writelines(debug_lines)

    # Parse the tool output for failures
    with open(tool_output_file, "r+") as f:
        lines = f.readlines()
        err_msg = next(
            (line for line in reversed(lines) if "DEBUG TOOL first run for" in line),
            None,
        )

        print("======================================================")
        if err_msg:
            layer = err_msg.split("for ")[1].split(", input")[0]
            second_run_str = f"DEBUG TOOL second run for {layer},"
            failed_layer = (
                f"Failed layer is {err_msg.split('for')[1]}"
                if second_run_str not in "".join(lines)
                else "No model layer has failed"
            )
            print(f"DEBUG TOOL \033[1m{failed_layer}\033[0m")
            print("======================================================")
            f.write(failed_layer + "\n")

            # Prepare repro code if failure detected
            if failed_layer != "No model layer has failed":
                if generate_repro_code_flag:
                    generate_repro_code_layer_debugging(err_msg, layer, model_path)
        else:
            print("No first run line found.")


def run_individual_layers(logfile, model_path, model_type):
    """Runs each layer of the model individually (in layer debugging mode).

    Args:
        logfile (str): Path to the complete model output log.
        model_path (str): Path to the model checkpoint.
        model_type (str): Type of model hf or fms.
    """    
    print("Running each layer individually........")
    layer_run = run_layers(model_path, model_type)
    command1 = ["python3", "-c", layer_run]
    # Show output in terminal as well as save in file
    with open(logfile, "a") as f:
        process = subprocess.Popen(
            command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in process.stdout:
            print(line, end="")
            f.write(line)


def run_model(
    model_type,
    model_path,
    tool_output_file,
    deepview_mode,
    generate_repro_code_flag,
    logfile="model_output.txt",
):
    """Main entry point to run a model and process its execution logs.

    Loads and compiles the model using `ModelHandler`, runs inference,
    and processes the output logs based on the active deepview mode.

    For `layer_debugging` mode, it inserts hooks and runs individual layers.
    For `unsupported_op` mode, it detects unsupported ops in the model.

    Args:
        model_type (str): Type of the model - hf or fms.
        model_path (str): Path to the model checkpoint.
        tool_output_file (str): Path to store filtered and analyzed output (e.g., unsupported ops or failed layers).
        deepview_mode (str): Mode to run DeepView in; can include 'layer_debugging' or 'unsupported_op'.
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
            model_handler = ModelHandler(
                model_type=model_type,
                model_path=model_path,
                prompt="What is the capital of India?",
            )
            model_handler.load_and_compile_model()
            model_handler.prep_input()

            print("Reached first infer call post compile.....")
            try:
                if "layer_debugging" in deepview_mode:
                    model_handler.insert_forward_hooks()

                model_handler.infer()

                if "layer_debugging" in deepview_mode:
                    model_handler.remove_forward_hooks()
                    with open("model_list.txt", "w") as file:
                        json.dump(
                            {k: list(v) for k, v in model_handler.layer_list.items()},
                            file,
                        )

                    run_individual_layers(logfile, model_path, model_type)

            except Exception as e:
                print(f"Exception occurred: {e}")

    # Process the logfile to create the tool_output_file
    if "unsupported_op" in deepview_mode:
        process_output_unsupported_ops(
            tool_output_file, logfile, generate_repro_code_flag
        )
    if "layer_debugging" in deepview_mode:
        process_output_layer_debugging(
            tool_output_file, logfile, generate_repro_code_flag, model_path
        )

    # Since both our modes produce outputs before this line, commenting out the code after that -- this can be enabled in future
    # Update lazyhandle after first run of inference
    # print("Updating lazyhandle")
    # torch_sendnn.update_lazyhandle()
