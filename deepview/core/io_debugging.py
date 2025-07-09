# Standard
from pathlib import Path
import numpy as np
import torch.nn as nn
from pycony import *
import torch_sendnn
import subprocess
import itertools
import inspect
import pickle
import shutil
import torch
import re
import os

# Local
from deepview.core.individual_layer_run_with_inputs import run_layers_with_inputs

def convert_attr_path_indexed(attr_path):
    pattern = re.compile(r'(\b\w+?)(\d+)\b')
    return pattern.sub(r'\1[\2]', attr_path)

def convert_attr_path(attr_path):
    attr_path = 'model.' + attr_path
    def replace_numeric_attr(match):
        number = match.group(1)
        tail = match.group(2)
        return f'[{number}]{tail}'
    pattern = re.compile(r'\.(\d+)(\.|$)')
    converted = pattern.sub(replace_numeric_attr, attr_path)
    return converted

def calc_output_diff(cpu_output_tensor, aiu_output_tensor, metrics):
    diff = {}
    if "abs_diff" in metrics:
        diff["abs_diff"] = torch.norm(torch.abs(cpu_output_tensor - aiu_output_tensor)).item()
    if "cos_sim" in metrics:
        cos = nn.CosineSimilarity(dim=-1)
        diff["cos_sim"] = (cos(cpu_output_tensor,aiu_output_tensor)).mean().item()
    return diff


def get_layer_thresholds(model_path, model_type):
    model_folder_name = model_path.replace("/", "--")
    thresholds_folder = "/home/senuser/a5-deepview/deepview/layerwise-thresholds"
    thesholds_filename = f"{model_folder_name}-thresholds.json"
    thresholds_filepath = os.path.join(thresholds_folder, model_folder_name, "model-forward", thesholds_filename)
    with open(thresholds_filepath, 'r') as f:
        thresholds_data = json.load(f)
    del thresholds_data["model_id"]
    merged_thresholds_data = {}
    for key, value_list in thresholds_data.items():
        merged_thresholds_data[key] = {}
        for item_dict in value_list:
            for k, v in item_dict.items():
                merged_thresholds_data[key][k] = v
    return merged_thresholds_data


def get_layerwise_outputs_cpu(model_handler):
    full_output_dict = {}
    for str_layer, _ in model_handler.layer_inputs.items():
        outputs = model_handler.layer_outputs[str_layer]
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = 'model'
        full_output_dict[sub_layer] = outputs
    return full_output_dict


def generate_layerwise_output_diffs(aiu_model_handler, cpu_layer_outputs, thresholds):
    model = aiu_model_handler.model

    metrics = list(thresholds.keys())
    layers_done = []
    print("Running each layer individually........")
    os.makedirs("temp", exist_ok=True)
    for str_layer, inputval in aiu_model_handler.layer_inputs.items():
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = 'model'
        if sub_layer in layers_done:
            continue
        if sub_layer != "model" and sub_layer != "model.base_model":
            tmp_filename = str_layer.replace(".", "_")
            torch.save(inputval, "temp/"+tmp_filename+"_input.pth")
            layer_run = run_layers_with_inputs(aiu_model_handler.model_path, sub_layer, tmp_filename)
            command1 = ["python3", "-c", layer_run]
            process = subprocess.run(
                command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")

            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW Layer is {sub_layer}. \n"
                )
            if process.returncode != 0:
                print(
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW \033[1mError running {sub_layer}\n\033[0m"
                    "DEEPVIEW========================================================================\n"
                )
                return sub_layer, 0
            else:
                result = torch.load("temp/"+tmp_filename+"_output_kwargs.pth")

                ## Calculate diff with CPU output
                diffs = calc_output_diff(cpu_layer_outputs[sub_layer], result, metrics)

                ## Compare with threshold
                key_in_thresholds_json = re.sub(r'\[(\d+)\]', r'\1', sub_layer)
                flag = False
                for metric in metrics:
                    if diffs[metric] > thresholds[metric][key_in_thresholds_json]:
                        flag = True
                        break
                if flag:
                    print(
                        f"DEEPVIEW Threshold test failed for {sub_layer}. Observed {metric} = {diffs[metric]}. Threshold = {thresholds[metric][key_in_thresholds_json]}.\n"
                         "DEEPVIEW========================================================================\n"
                    )
                    return sub_layer, 1
                else:
                    print(
                        f"DEEPVIEW Threshold test passed for {sub_layer}. Observed {metric} = {diffs[metric]}. Threshold = {thresholds[metric][key_in_thresholds_json]}.\n"
                         "DEEPVIEW========================================================================\n"
                    )
                    layers_done.append(sub_layer)
    shutil.rmtree("temp")
    return None, 2




