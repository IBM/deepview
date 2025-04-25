import os
import sys
import subprocess
import argparse
import shutil
import torch_sendnn

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.model_runner import run_model
from core.hook_monitor import enable_unsupported_op_mode, clear_unsupported_op_mode


parser = argparse.ArgumentParser(
    description="Script to run DeepView tool on any model."
)

parser.add_argument(
    '--model_type', 
    choices=['fms','hf'],
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
        '--mode',
        nargs='+',
        choices=['unsupported_op'],
        default=['unsupported_op'],
        help="Modes: [unsupported_op] (Choose one or more)"
)

parser.add_argument(
    '--show_details',
    action='store_true',
    help="Print stack trace and other details, valid only with unsupported_op."
)

parser.add_argument(
    '--output_file',
    default="debug_tool_log.txt",
    required=True,
    help="Print stack trace and other details, valid only with unsupported_op."
)

args = parser.parse_args()


if args.show_details and 'unsupported_op' not in args.mode:
    print("Error: --show_details can only be used if 'unsupported_op' is specified in --mode.")
    sys.exit(1)


for mode in args.mode:
    if mode == 'unsupported_op':
        enable_unsupported_op_mode(args.show_details)
        

# ====================== This block will be removed if changes are merged to backends.py ======================
# Find local installation folder of torch_sendnn
installed_sendnn_folder = os.path.dirname(torch_sendnn.__file__)
new_sendnn_folder = os.path.join(os.getcwd(),'torch_sendnn')
shutil.copytree(installed_sendnn_folder, new_sendnn_folder, dirs_exist_ok=True)


# Modify backends.py
original_file_path = os.path.join(new_sendnn_folder, 'backends.py')
new_backends_file_path = 'core/backends.py'
shutil.copy2(new_backends_file_path, original_file_path)


# Append torch_sendnn to python path
os.environ["PYTHONPATH"] = os.getcwd() + os.pathsep + os.environ["PYTHONPATH"]
print(os.environ["PYTHONPATH"])
# ==============================================================================================================

# Run model and save output
run_model(args.model_type, args.model, args.output_file)

# Tear down the environment 
clear_unsupported_op_mode()
shutil.rmtree(new_sendnn_folder)