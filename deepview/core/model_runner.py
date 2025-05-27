# Allows running specific modules/layers in isolation with preserved context
import torch
import re
import sys
import os
import json
import subprocess
from torch_sendnn import torch_sendnn

from deepview.core.generate_minimal_repro import generate_repro_code_unsupported_ops,  generate_repro_code_layer_debugging
from deepview.utils.model_handler import ModelHandler

class PrintOutput:
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a')
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return self.stream.isatty()
    def fileno(self):
        return self.stream.fileno()
    def close(self):
        self.file.close()


def set_environment():
    old_output_file = 'model_output.txt'
    if os.path.exists(old_output_file):
        print("Deleting the old model_output.txt..............")
        os.remove(old_output_file)
    os.environ['DTLOG_LEVEL'] = 'error'
    os.environ['TORCH_SENDNN_LOG'] = 'CRITICAL'
    os.environ['DT_DEEPRT_VERBOSE'] = '-1'
    os.environ['PYTHONUNBUFFERED'] = '1'


def process_output_unsupported_ops(tool_output_file, logfile, generate_repro_code_flag):
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.
    unknown_nodes = []
    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        for line in infile:
            if line.startswith("DEBUG TOOL"):
                outfile.write(line)
                match = re.search(r'DEBUG TOOL Caught error for (.*?):', line)
                if match:
                    node = match.group(1)
                    unknown_nodes.append(node)

        strip = lambda s: re.sub(r'\x1b\[[0-9;]*m', '', s)
        seen = set()
        unique_unknown_nodes = [op for op in unknown_nodes if not re.match(r'.*_\d+$', strip(op)) and (strip(op) not in seen and not seen.add(strip(op)))]
        if len(unique_unknown_nodes) == 0:
            no_unsup_op = (
                "DEBUG TOOL========================================================================\n"
                "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n"
            )
            print(no_unsup_op)
            outfile.write(no_unsup_op)
        else:
            unknown_nodes_str = '\n'.join(sorted(unique_unknown_nodes))
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

def process_output_layer_debugging(tool_output_file, logfile, generate_repro_code_flag, model_path):
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.
    
    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        debug_lines = [line for line in infile if line.startswith("DEBUG TOOL")]
        outfile.writelines(debug_lines)

    # Parse the tool output for failures
    with open(tool_output_file, "r+") as f:
        lines = f.readlines()
        err_msg = next((line for line in reversed(lines) if "DEBUG TOOL first run for" in line), None)

        print("======================================================")
        if err_msg:
            layer = err_msg.split("for ")[1].split(", input")[0]
            second_run_str = f"DEBUG TOOL second run for {layer},"
            failed_layer = f"Failed layer is {err_msg.split('for')[1]}" if second_run_str not in ''.join(lines) else "No model layer has failed"
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
    print("Running each layer individually........")
    command1 = [
        'python3', 'deepview/deepview/core/test_layers.py', '--model_path', model_path, '--model_type', model_type,
    ]

    # Show output in terminal as well as save in file
    with open(logfile, "a") as f:
        process = subprocess.Popen(command1,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
             print(line, end='')
             f.write(line)

def run_model(model_type, model_path, tool_output_file, deepview_mode, generate_repro_code_flag, logfile='model_output.txt'):
    if generate_repro_code_flag:
        torch_sendnn.preserve_lazyhandle()

    model_handler = ModelHandler(model_type=model_type, model_path=model_path, prompt='What is the capital of India?')
    model_handler.load_and_compile_model()
    model_handler.prep_input()

    # Prints generated by compile_and_infer are shown in terminal as well as saved in logfile.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = PrintOutput(logfile, original_stdout)
    tee_stderr = PrintOutput(logfile, original_stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    print("Reached first infer call post compile.....")
    try:
        if 'layer_debugging' in deepview_mode:
            model_handler.insert_forward_hooks()

        model_handler.infer()

        if 'layer_debugging' in deepview_mode:
            model_handler.remove_forward_hooks()
            with open("model_list.txt", "w") as file:
                json.dump({k: list(v) for k, v in model_handler.layer_list.items()}, file)
                file.close()
            run_individual_layers(logfile, model_path, model_type)

    except Exception as e:
        print(f"Exception occurred: {e}", file=original_stderr)
    finally:
        tee_stdout.flush()
        tee_stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()

    # Process the logfile to create the tool_output_file 
    if 'unsupported_op' in deepview_mode:
        process_output_unsupported_ops(tool_output_file, logfile, generate_repro_code_flag)
    if 'layer_debugging' in deepview_mode:
        process_output_layer_debugging(tool_output_file, logfile, generate_repro_code_flag, model_path)

    # Since both our modes produce outputs before this line, commenting out the code after that -- this can be enabled in future
    # Update lazyhandle after first run of inference
    # print("Updating lazyhandle")
    # torch_sendnn.update_lazyhandle()
