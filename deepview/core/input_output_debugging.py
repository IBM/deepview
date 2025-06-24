# Standard
from pathlib import Path
from pycony import *
import subprocess
import itertools
import inspect
import pickle
import torch
import re

# Local
from deepview.core.individual_layer_run_with_inputs import run_layers_with_inputs


def convert_attr_path(attr_path):
    attr_path = 'model.' + attr_path
    def replace_numeric_attr(match):
        number = match.group(1)
        tail = match.group(2)
        return f'[{number}]{tail}'
    pattern = re.compile(r'\.(\d+)(\.|$)')
    converted = pattern.sub(replace_numeric_attr, attr_path)
    return converted

def generate_individual_layer_output(model_handler, model_path, model_type, device_to_run, timestamp):
    """Generates layer outputs by running each layer of the model individually on the inputs collected in forward pass.

    Iterates over the all layers and captures the outputs of the layers
    At the end of the run, stores the inputs and outputs per layer in a pkl file.

    Args:
        model_path (str): Path to the model checkpoint directory.
        model_type (str): Model type, either 'hf' (HuggingFace) or 'fms' (Foundation Model Stack).
        layer_list (dict): Dictionary mapping layer/module names to a set containing input shape and data type.
    """
    failed_layer = "No failed layer"

    torch.compiler.reset()
    torch._dynamo.reset()
    model = model_handler.model

    if device_to_run == 'aiu':
        input_outputs = []
        layers_done = []
        print("Running each layer individually........")
        filename = f"AIU_run_{timestamp}.pkl"
        for str_layer, inputval in model_handler.layer_inputs.items():
            if str_layer:
                sub_layer = convert_attr_path(str_layer)
            else:
                sub_layer = 'model'
            if sub_layer in layers_done:
                continue
            if sub_layer != "model" and sub_layer != "model.base_model":
                print("Layer is ", sub_layer)
                
                target_layer = eval(sub_layer)
                forward_signature = inspect.signature(target_layer.forward)
                expected_args = list(forward_signature.parameters.keys())
                print(f"Expected Arguments: {expected_args}")
                
                print("Inputs collected:", inputval)
                inputvals = list(inputval)
                for i, val in enumerate(inputvals):
                    if isinstance(val, torch.Tensor):
                        print(f"  [{i}]: Tensor of shape {val.shape}")
                    else:
                        print(f"  [{i}]: {val}")

                if len(inputval) < len(expected_args):
                    zipped_inputs = list(itertools.zip_longest(expected_args, inputval, fillvalue=None))
                else:
                    zipped_inputs = list(zip(expected_args, inputval))
                
                kwargs = dict(zipped_inputs)
                torch.save(kwargs, "input_kwargs.pth")

                print(
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW Running {sub_layer} with input {kwargs}"
                )

                layer_run = run_layers_with_inputs(model_path, sub_layer)
                command1 = ["python3", "-c", layer_run]
                process = subprocess.run(
                    command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )

                for line in process.stdout:
                    print(line, end="")
                if process.returncode != 0:
                    print(
                        "DEEPVIEW========================================================================\n"
                        f"DEEPVIEW \033[1mError running {sub_layer}\n\033[0m"
                        "DEEPVIEW========================================================================\n"
                    )
                    failed_layer = sub_layer
                    break
                else:
                    result = torch.load("output_kwargs.pth")
                    print(
                        f"DEEPVIEW Successfully ran {sub_layer}\n"
                        "DEEPVIEW========================================================================\n"
                    )
                    
                    input_output_dict = {}
                    input_output_dict['layer'] = sub_layer
                    input_output_dict['input'] = kwargs
                    input_output_dict['output'] = result
                    input_outputs.append(input_output_dict)

                layers_done.append(sub_layer)

        with open(filename, 'wb') as f:
            pickle.dump(input_outputs, f) 

    elif device_to_run == 'cpu':
        input_outputs = []
        filename = f"CPU_run_{timestamp}.pkl"
        for str_layer, inputval in model_handler.layer_inputs.items():
            if str_layer:
                sub_layer = convert_attr_path(str_layer)
            else:
                sub_layer = 'model'

            inputs = model_handler.layer_inputs[str_layer]
            outputs = model_handler.layer_outputs[str_layer]

            input_output_dict = {}
            input_output_dict['layer'] = sub_layer
            input_output_dict['input'] = inputs
            input_output_dict['output'] = outputs
            input_outputs.append(input_output_dict)

        with open(filename, 'wb') as f:
            pickle.dump(input_outputs, f) 

    return input_outputs




