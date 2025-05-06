import os
import sys
import argparse



project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.model_runner import run_model, set_environment
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
        choices=['unsupported_op', 'layer_debugging'],
        default=['unsupported_op'],
        help="Modes: [unsupported_op, layer_debugging] (Choose one or more)"
)

parser.add_argument(
    '--show_details',
    action='store_true',
    help="Print stack trace and other details, valid only with unsupported_op."
)

parser.add_argument(
    '--generate_repro_code',
    action='store_true',
    help="Generate minimal reproducible code for unsupported operation."
)

parser.add_argument(
    '--output_file',
    default="debug_tool_log.txt",
    required=True,
    help="Name of the file in which the debug tool output will be stored."
)

args = parser.parse_args()

if args.model_type == 'hf':
    print("Support for HF models yet to be implemented")
    sys.exit()

for mode in args.mode:
    if mode == 'unsupported_op':
        enable_unsupported_op_mode(args.show_details)

# Setting the environment variables
set_environment()

# Run the model
print("Running the model")
run_model(args.model, args.output_file, args.generate_repro_code)
print("Model run completed")

# Tear down the environment 
clear_unsupported_op_mode()
