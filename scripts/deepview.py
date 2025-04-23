import os
import subprocess
import sys
import argparse

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
    "--sendnn_path",
    type=str,
    default="torch_sendnn",
    help="Path to torch_sendnn package folder",
)

parser.add_argument(
    '--show_stack_trace',
    action='store_true',
    help="Print stack trace with unsupported op."
)

subparsers = parser.add_subparsers(
    dest='model_type', 
    required=True, 
    help='The type of model you want to debug.'
)

parser_hf = subparsers.add_parser('hf', help='For running HF models directly')
parser_hf.add_argument(
    '--model_name', 
    required=True, 
    help='Model name in HF format'
)

parser_fms = subparsers.add_parser('fms', help='For running FMS models')
parser_fms.add_argument(
    '--model_path', 
    required=True, 
    help='Path to the FMS model'
)



args = parser.parse_args()

convert_unknown_lineno = (543,556)
original_file_path = os.path.join(args.sendnn_path, "torch_sendnn", "backends.py")
insert_file_path = '../core/convert_unknown_func.txt'
if args.show_stack_trace:
    insert_file_path = '../core/convert_unknown_func_with_stack_trace.txt'
modify_file(original_file_path, insert_file_path, convert_unknown_lineno[0], convert_unknown_lineno[1])



# lazy_handles_lineno = (1998,2002)


# Reinstall torch_sendnn library 
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=args.sendnn_path)
# Append torch_sendnn to python path
sys.path.insert(0, args.sendnn_path)


if args.model_type == 'hf':
    command = [
        'python3', '../core/inference_hf.py',
        '--model_name', args.model_name,
    ]
elif args.model_type == 'fms':
    command = [
        'python3', '../core/inference_fms.py',
        '--architecture', 'hf_pretrained',
        '--model_path', args.model_path,
        '--tokenizer', args.model_path, 
        '--device_type', 'aiu' 
        '--unfuse_weights', 
        '--compile', 
        '--compile_dynamic',
        '--default_dtype','fp16',
        '--fixed_prompt_length','64', 
        '--max_new_tokens','20', 
        '--timing','per-token ',
        '--batch_size','1'
    ]


  
with open('model_output.txt', 'w') as f:
    subprocess.run(
        command,
        stdout=f,    # ✅ stdout redirected to file
        stderr=subprocess.STDOUT   # ✅ stderr merged into stdout
    )

