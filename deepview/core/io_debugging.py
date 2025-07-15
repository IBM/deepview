# Standard
import torch.nn.functional as F
import torch.nn as nn
from pycony import *
import subprocess
import pickle
import shutil
import torch
import json
import re
import os

# Local
from aiu_fms_testing_utils.utils.metrics_utils import tensor_cos_sim, tensor_abs_diff, abs_diff_linalg_norm, list_mean
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

def replace_zeros(tensor, eps=1e-8):
    norm = tensor.norm(dim=-1, keepdim=True)
    mask = norm < eps
    tensor[mask] = torch.randn_like(tensor[mask]) * eps
    return tensor

def calc_output_diff(cpu_output_tensor, aiu_output_tensor, metric):
    if metric == "abs_diff":
        return abs_diff_linalg_norm(tensor_abs_diff(cpu_output_tensor,aiu_output_tensor).numpy())
    elif metric == "cos_sim_avg" or metric == "cos_sim_mean":
        cos_sim = (tensor_cos_sim(cpu_output_tensor, aiu_output_tensor)).numpy()
        return list_mean(cos_sim)

def is_acceptable(obs, thresh):
    atol = float(os.getenv("DEEPVIEW_ABS_TOLERANCE", 1e-6))
    rtol = float(os.getenv("DEEPVIEW_REL_TOLERANCE", 0.05))
    if abs(obs - thresh) <= (rtol * thresh + atol):
        return True
    return False



def get_layer_thresholds(model_path, model_type):
    model_folder_name = model_path.replace("/", "--")
    thresholds_folder = os.getenv('DEEPVIEW_THRESHOLDS_FOLDERPATH')
    thesholds_filename = f"{model_folder_name}-thresholds.json"
    thresholds_filepath = os.path.join(thresholds_folder, model_folder_name, "generate", thesholds_filename)
    with open(thresholds_filepath, 'r') as f:
        thresholds_data = json.load(f)
    del thresholds_data["model_id"]
    return thresholds_data


def get_layerwise_outputs_cpu(model_handler):
    full_output_dict = {}
    for str_layer, output in model_handler.layer_outputs.items():
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = 'model'
        full_output_dict[sub_layer] = output
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
            with open("temp/"+tmp_filename+"_input.pkl", 'wb') as f:
                pickle.dump(inputval, f) 
            layer_run = run_layers_with_inputs(aiu_model_handler.model_path, sub_layer, tmp_filename)
            command1 = ["python3", "-c", layer_run]
            process = subprocess.run(
                command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")

            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW Layer is {sub_layer}."
                )
            if process.returncode != 0:
                print(
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW \033[1mError running {sub_layer}\n\033[0m"
                    "DEEPVIEW========================================================================\n"
                )
                return sub_layer, 0
            else:
                with open("temp/"+tmp_filename+"_output_kwargs.pkl", 'rb') as f:
                    result = pickle.load(f) 
                key_in_thresholds_json = re.sub(r'\[(\d+)\]', r'\1', sub_layer)
                count = 0
                for metric in metrics:
                    observed_diff = calc_output_diff(cpu_layer_outputs[sub_layer], result, metric)
                    threshold_diff = thresholds[metric][key_in_thresholds_json]
                    print(f"DEEPVIEW Metric: {metric}. Observed Value = {observed_diff}. Threshold = {threshold_diff}.")
                    if ((metric == 'abs_diff') and (observed_diff > threshold_diff)) or (((metric == 'cos_sim_avg') or (metric == 'cos_sim_mean')) and (observed_diff < threshold_diff)) :
                        if not is_acceptable(observed_diff,threshold_diff):
                            count = count + 1
                if count > 0:
                    print(
                        f"DEEPVIEW Threshold test failed for {sub_layer}.\n"
                        "DEEPVIEW========================================================================\n"
                    )
                    return sub_layer, 1
                print(
                    f"DEEPVIEW Threshold test passed for {sub_layer}.\n"
                    "DEEPVIEW========================================================================\n"
                )
                layers_done.append(sub_layer)
    shutil.rmtree("temp")
    return None, 2




