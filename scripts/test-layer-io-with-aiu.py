#!/usr/bin/env python3

import torch
from torch_sendnn import torch_sendnn
from fms.models import get_model
import pprint
import os
from pycony import * 

os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

from deepview.utils.io_utils import * 

# Path to the model artifcats
MODEL_PATH='/mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct'

torch.set_grad_enabled(False)
# torch._dynamo.config.automatic_dynamic_shapes = False 
# torch._dynamo.config.dynamic_shapes = False 

model = get_model(
    architecture='hf_pretrained',
    variant=None,
    model_path=MODEL_PATH,
    device_type="cpu",
    data_type=torch.float16,
    source=None,
    distributed_strategy=None,
    group=None,
    linear_config={'linear_type': 'torch_linear'},
    fused_weights=False,
)

# model.base_model.layers = model.base_model.layers[:1]

print('Model is loaded')


layers_stack = load_layer_stack()
for layer_name, save_dict in layers_stack:
    if layer_name not in ('model', 'model.base_model'):
        print(f"layer_name:{layer_name} --> dir: {save_dict['save_dir']}")

print(40*'=')
print("Compile and Run a layer manually as following:")
print("e.g. to test `model.base_model.layers[0].ln` pass `model` handle and its dir as argument: \ncompile_and_run_layer(model, 'model_base_model_layers_0_ln_0' )")
print(40*'=')
open_console()

# Loop over layers 

for layer_name, save_dict in layers_stack:
    if layer_name not in ('model', 'model.base_model'):
        print(40*'=')
        print(f"Now Testing layer {layer_name} ios from dir: {save_dict['save_dir']}")
        print(40*'=')
        try:
            compile_and_run_layer(model, save_dict['save_dir'])
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
        print(f'TESTED: {layer_name}')


exit(1)
