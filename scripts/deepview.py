import os
import subprocess
import sys
import argparse
import shutil
import torch_sendnn



def modify_file(original_file, insert_file, x, y):
    with open(original_file, 'r') as f:
        original_lines = f.readlines()
    with open(insert_file, 'r') as f:
        insert_lines = f.readlines()
    updated_lines = original_lines[:x-1] + insert_lines + original_lines[y:]
    with open(original_file, 'w') as f:
        f.writelines(updated_lines)


parser = argparse.ArgumentParser(
    description="Script to run DeepView tool on any model."
)

parser.add_argument(
    '--model_type', 
    choices=['fms' 'hf'],
    required=True, 
    default='fms',
    help='The type of model you want to debug - fms or hf.'
)

parser.add_argument(
    '--model', 
    required=True, 
    help='Model name in HF format or model path'
)

parser.add_argument(
    '--show_details',
    action='store_true',
    help="Print stack trace and other details with unsupported op."
)


args = parser.parse_args()


## USE AST
convert_unknown_lineno = (543,556)
sendnn_folder = os.path.dirname(torch_sendnn.__file__)
dst_folder = os.path.join(os.getcwd(), os.path.basename(sendnn_folder))
shutil.copytree(sendnn_folder, dst_folder)

original_file_path = os.path.join(dst_folder, 'torch_sendnn', 'backends.py')
insert_file_path = '../core/function_modifications/convert_unknown_func.txt'
if args.show_details:
    insert_file_path = '../core/function_modifications/convert_unknown_func_with_stack_trace.txt'
modify_file(original_file_path, insert_file_path, convert_unknown_lineno[0], convert_unknown_lineno[1])

## Add more information (layers etc)

# lazy_handles_lineno = (1998,2002)


# Reinstall torch_sendnn library 
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=dst_folder)
# Append torch_sendnn to python path
sys.path.insert(0, dst_folder)


if args.model_type == 'hf':
    command = [
        'python3', '../core/inference_hf.py',
        '--model_name', args.model,
    ]
elif args.model_type == 'fms':
    command = [
        'python3', '../core/inference_fms.py',
        '--architecture', 'hf_pretrained',
        '--model_path', args.model,
        '--tokenizer', args.model, 
        '--device_type', 'aiu',
        '--unfuse_weights', 
        '--compile', 
        '--compile_dynamic',
        '--default_dtype','fp16',
        '--fixed_prompt_length','64', 
        '--max_new_tokens','20', 
        '--timing','per-token',
        '--batch_size','1'
    ]


with open("model_output.txt", "w") as f:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in process.stdout:
        print(line, end='')  # ✅ print to terminal
        f.write(line)        # ✅ write to file
    process.wait()


input_file = "model_output.txt"
output_file = "debug_tool_output.txt"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        if line.startswith("DEBUG TOOL"):
            outfile.write(line)

## If the code breaks, still should output 
## Show output in terminal as well as save in file
## pipe debug tool output to another file

## Handle case: All ops are supported 