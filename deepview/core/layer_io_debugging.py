# Standard
import json
import os
import pickle
import re
import shutil
import subprocess

from deepview.utils.model_handler import extract_hf_model_id

# Third Party
from aiu_fms_testing_utils.utils.metrics_utils import (
    abs_diff_linalg_norm,
    list_mean,
    tensor_abs_diff,
    tensor_cos_sim,
)

# Local
from deepview.core.aiu_input_capture import run_model_for_inputs
from deepview.core.individual_layer_run_with_inputs import run_layers_with_inputs

# Defining some constants
SUCCESS = 2
THRESHOLD_TEST_FAILED = 1
LAYER_RUN_FAILED = 0


def convert_attr_path(attr_path):
    """Converts the name of the modules to match the format in thresholds file."""
    attr_path = "model." + attr_path

    def replace_numeric_attr(match):
        number = match.group(1)
        tail = match.group(2)
        return f"[{number}]{tail}"

    pattern = re.compile(r"\.(\d+)(\.|$)")
    converted = pattern.sub(replace_numeric_attr, attr_path)
    return converted


def get_thresholds_json_file(model_path):
    """
    Attempts to locate the threshold JSON file for a model by:
    1. Extracting the HF model ID and checking in DEEPVIEW_THRESHOLDS_FOLDERPATH
    2. Falling back to dynamic subfolder structure with 'generate' suffix
    """
    thresholds_folder = os.getenv("DEEPVIEW_THRESHOLDS_FOLDERPATH", ".")
    if not thresholds_folder:
        raise EnvironmentError("DEEPVIEW_THRESHOLDS_FOLDERPATH is not set")

    model_id = extract_hf_model_id(model_path)

    # First attempt: direct match using model_id inside 'generate' subfolder
    direct_folder = os.path.join(thresholds_folder, f"{model_id}", "generate")
    if os.path.isdir(direct_folder):
        for fname in os.listdir(direct_folder):
            if fname.endswith(".json"):
                return os.path.join(direct_folder, fname)

    # Fallback: construct path using model folder name
    if model_path.count("/") > 1:
        model_folder_name = model_path.split("/")[-2] + "--" + model_path.split("/")[-1]
    else:
        model_folder_name = model_path.replace("/", "--")

    fallback_folder = os.path.join(thresholds_folder, model_folder_name, "generate")
    if os.path.isdir(fallback_folder):
        for fname in os.listdir(fallback_folder):
            if fname.endswith(".json"):
                return os.path.join(fallback_folder, fname)

    raise FileNotFoundError("No threshold JSON file found.")


def calc_output_diff(cpu_output_tensor, aiu_output_tensor, metric):
    """Calculates the diff of outputs between the CPU and AIU runs."""
    if metric == "abs_diff":
        return abs_diff_linalg_norm(
            tensor_abs_diff(cpu_output_tensor, aiu_output_tensor).numpy()
        )
    elif metric == "cos_sim_avg" or metric == "cos_sim_mean":
        cos_sim = (tensor_cos_sim(cpu_output_tensor, aiu_output_tensor)).numpy()
        return list_mean(cos_sim)


def is_acceptable(obs, thresh):
    """Checks if the observed diff is within the specified range with respect to threshold diffs."""
    atol = float(os.getenv("DEEPVIEW_ABS_TOLERANCE", 1e-6))
    rtol = float(os.getenv("DEEPVIEW_REL_TOLERANCE", 0.05))
    return abs(obs - thresh) <= (rtol * thresh + atol)


def get_layer_thresholds(thresholds_filepath):
    """Gets the layerwise thresdholds from the thresholds json file."""
    with open(thresholds_filepath, "r") as f:
        thresholds_data = json.load(f)
    del thresholds_data["model_id"]
    return thresholds_data


def get_layerwise_outputs_cpu(model_handler):
    """Gets the output of CPU run in dict format with properly formatted keys."""
    full_output_dict = {}
    for str_layer, output in model_handler.layer_outputs.items():
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = "model"
        full_output_dict[sub_layer] = output
    return full_output_dict


def generate_layerwise_inputs_aiu(
    model_type, model_path, deepview_mode, layer_inputs_file
):
    layer_inputs = None
    model_run = run_model_for_inputs(
        model_type, model_path, deepview_mode, layer_inputs_file
    )
    command1 = ["python3", "-c", model_run]
    process = subprocess.run(
        command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="")
    print(
        "DEEPVIEW========================================================================\n"
        f"DEEPVIEW Input capture for {model_path} ran succesfully.\n"
        "DEEPVIEW========================================================================\n"
    )
    if process.returncode != 0:
        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW \033[1mError running {model_path}\n\033[0m\n"
            "DEEPVIEW========================================================================\n"
        )
    with open(layer_inputs_file, "rb") as f:
        layer_inputs = pickle.load(f)
    return layer_inputs


def generate_layerwise_output_diffs(
    aiu_model_handler, inputs_filename, cpu_layer_outputs, thresholds
):
    """Runs the model on AIU layer-by-layer, meaures the output divergence at each layer, and compares with the thresholds.

    Returns status code 2 if all tests pass, 1 and the offernding layer if the threshold test fails for any particular layer,
    and 0 and the offernding layer if the layer run fails.
    """
    model = aiu_model_handler.model

    metrics = list(thresholds.keys())
    layers_done = []
    print("Running each layer individually........")
    os.makedirs("dv_layer_io_debugging_tmp", exist_ok=True)
    for str_layer in aiu_model_handler.layer_inputs.keys():
        if str_layer:
            sub_layer = convert_attr_path(str_layer)
        else:
            sub_layer = "model"
        if sub_layer in layers_done:
            continue
        if sub_layer != "model" and sub_layer != "model.base_model":
            layer_run = run_layers_with_inputs(
                aiu_model_handler.model_path, sub_layer, str_layer, inputs_filename
            )
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
                return sub_layer, LAYER_RUN_FAILED
            else:
                with open(
                    "dv_layer_io_debugging_tmp/" + str_layer + "_output_kwargs.pkl",
                    "rb",
                ) as f:
                    result = pickle.load(f)
                key_in_thresholds_json = re.sub(r"\[(\d+)\]", r"\1", sub_layer)
                count = 0
                for metric in metrics:
                    observed_diff = calc_output_diff(
                        cpu_layer_outputs[sub_layer], result, metric
                    )
                    threshold_diff = thresholds[metric][key_in_thresholds_json]
                    print(
                        f"DEEPVIEW Metric: {metric}. Observed Value = {observed_diff}. Threshold = {threshold_diff}."
                    )
                    if (
                        (metric == "abs_diff") and (observed_diff > threshold_diff)
                    ) or (
                        ((metric == "cos_sim_avg") or (metric == "cos_sim_mean"))
                        and (observed_diff < threshold_diff)
                    ):
                        if not is_acceptable(observed_diff, threshold_diff):
                            count = count + 1
                if count > 0:
                    print(
                        f"DEEPVIEW Threshold test failed for {sub_layer}.\n"
                        "DEEPVIEW========================================================================\n"
                    )
                    return sub_layer, THRESHOLD_TEST_FAILED
                print(
                    f"DEEPVIEW Threshold test passed for {sub_layer}.\n"
                    "DEEPVIEW========================================================================\n"
                )
                layers_done.append(sub_layer)
    shutil.rmtree("dv_layer_io_debugging_tmp")
    return None, SUCCESS
