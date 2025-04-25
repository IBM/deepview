import os
import sys
import subprocess
import argparse
import shutil

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
        required=True,
        help="Modes: [unsupported_op] (Choose one or more)"
)

parser.add_argument(
    '--show_details',
    action='store_true',
    help="Print stack trace and other details, valid only with unsupported_op."
)

args = parser.parse_args()


if not args.mode:
    run_model(args.model_type, args.model)
    sys.exit(1)

if args.show_details and 'A' not in args.mode:
    print("Error: --show_details can only be used if 'unsupported_op' is specified in --mode.")
    sys.exit(1)


for mode in args.mode():
    if mode == 'unsupported_op':
        enable_unsupported_op_mode(args.show_details)
        

# ====================== This block will be removed if changes are merged to backends.py ======================
# Modify backends.py
sendnn_folder = '/tmp/torch_sendnn'
original_file_path = os.path.join(sendnn_folder, 'torch_sendnn/backends.py')
new_backends_file_path = '../core/backends.py'
shutil.copy2(new_backends_file_path, original_file_path)

# Reinstall torch_sendnn library 
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', '.'], cwd=sendnn_folder)

# Append torch_sendnn to python path
os.environ["PYTHONPATH"] = '/tmp/torch_sendnn' + os.pathsep + os.environ["PYTHONPATH"]
print(os.environ["PYTHONPATH"])
# ==============================================================================================================


# Run model and save output
run_model(args.model_type, args.model)

# Tear down the environment 
clear_unsupported_op_mode()

