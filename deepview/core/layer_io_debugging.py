# Standard
import json
import os
import pickle
import re
import shutil
import subprocess
import sys

# Third Party
from aiu_fms_testing_utils.utils.metrics_utils import (
    abs_diff_linalg_norm,
    list_mean,
    tensor_abs_diff,
    tensor_cos_sim,
)
import torch

# Local
from deepview.core.aiu_input_capture import run_model_for_inputs
from deepview.core.individual_layer_run_with_inputs import run_layers_with_inputs
from deepview.utils.ModelHandler.model_handler_utils import (
    create_model_handler,
    extract_hf_model_id,
    setup_model_handler,
)

# Defining some constants
SUCCESS = 2
THRESHOLD_TEST_FAILED = 1
LAYER_RUN_FAILED = 0


def get_thresholds_json_file(model_path):
    """
    Attempts to locate the threshold JSON file for a model by:
    1. Extracting the HF model ID and checking in DEEPVIEW_THRESHOLDS_FOLDERPATH
    2. Falling back to dynamic subfolder structure with 'generate' suffix
    """
    print(f"Getting thresholds JSON file for {model_path}.....")
    thresholds_folder = os.getenv("DEEPVIEW_THRESHOLDS_FOLDERPATH", ".")
    print(f"Using thresholds folder: {thresholds_folder}")
    if not thresholds_folder:
        raise EnvironmentError("DEEPVIEW_THRESHOLDS_FOLDERPATH is not set")

    model_id = extract_hf_model_id(model_path)
    print(f"Extracted model ID: {model_id}")
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
    if isinstance(cpu_output_tensor, tuple) and isinstance(aiu_output_tensor, tuple):
        if len(cpu_output_tensor) != len(aiu_output_tensor):
            raise ValueError(
                "Tuples must be of the same length for elementwise subtraction."
            )
        cpu_output_tensor = torch.cat([t.flatten() for t in cpu_output_tensor])
        aiu_output_tensor = torch.cat([t.flatten() for t in aiu_output_tensor])
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
    for layer, output in model_handler.layer_outputs.items():
        full_output_dict[layer] = output
    return full_output_dict


def generate_layerwise_inputs_aiu(model_type, model_path, layer_inputs_file):
    """Generates inputs per layer for AIU run by using hooks."""
    layer_inputs = None
    model_run = run_model_for_inputs(model_type, model_path, layer_inputs_file)
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
        return None
    else:
        with open(layer_inputs_file, "rb") as f:
            layer_inputs = pickle.load(f)
        return layer_inputs


def generate_layerwise_output_diffs(
    aiu_model_handler, inputs_filename, cpu_layer_outputs, thresholds
):
    """Runs the model on AIU layer-by-layer, measures the output divergence at each layer, and compares with the thresholds.

    Returns status code 2 if all tests pass, 1 and the offernding layer if the threshold test fails for any particular layer,
    and 0 and the offernding layer if the layer run fails.
    """
    model = aiu_model_handler.model

    metrics = list(thresholds.keys())
    layers_done = []
    print("Running each layer individually........")
    os.makedirs("dv_layer_io_debugging_tmp", exist_ok=True)
    for layer in aiu_model_handler.layer_inputs.keys():
        if layer in layers_done:
            continue
        if layer != "model" and layer != "model.base_model":
            layer_run = run_layers_with_inputs(
                aiu_model_handler.model_path, layer, inputs_filename
            )
            print(
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW Running layer {layer}:"
            )
            command1 = ["python3", "-c", layer_run]
            process = subprocess.run(
                command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")
            if process.returncode != 0:
                print(
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW \033[1mError running {layer}\n\033[0m"
                    "DEEPVIEW========================================================================\n"
                )
                return layer, LAYER_RUN_FAILED
            else:
                with open(
                    "dv_layer_io_debugging_tmp/" + layer + "_output_kwargs.pkl",
                    "rb",
                ) as f:
                    result = pickle.load(f)
                key_in_thresholds_json = re.sub(r"\[(\d+)\]", r"\1", layer)
                count = 0
                for metric in metrics:
                    observed_diff = calc_output_diff(
                        cpu_layer_outputs[layer], result, metric
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
                        f"DEEPVIEW Threshold test failed for {layer}.\n"
                        "DEEPVIEW========================================================================\n"
                    )
                    return layer, THRESHOLD_TEST_FAILED
                print(
                    f"DEEPVIEW Threshold test passed for {layer}.\n"
                    "DEEPVIEW========================================================================\n"
                )
                layers_done.append(layer)
    shutil.rmtree("dv_layer_io_debugging_tmp")
    return None, SUCCESS


def get_thresholds(model_path):
    """Returns the thresholds from the environment variable DEEPVIEW_THRESHOLDS_FILEPATH."""
    print(f"Getting thresholds for {model_path}.....")
    threshold_filepath = get_thresholds_json_file(model_path)
    if not threshold_filepath:
        print(f"Unable to find thresholds for {model_path}.")
        sys.exit(0)
    print("Getting layer output thresholds.....")
    thresholds = get_layer_thresholds(threshold_filepath)
    return thresholds


def get_layerwise_outputs(model_handler):
    model_handler.get_layer_io()
    layer_outputs = get_layerwise_outputs_cpu(model_handler)
    model_handler.remove_forward_hooks()
    model_handler.clear_layer_io()
    return layer_outputs


def get_layerwise_inputs(model_type, model_path, inputs_filename):
    """Generates inputs per layer for AIU run by using hooks."""
    layer_inputs = generate_layerwise_inputs_aiu(
        model_type, model_path, inputs_filename
    )
    if not layer_inputs:
        print(
            f"Input capture failed for {model_path} as model did not run successfully on AIU."
        )
        sys.exit(0)

    return layer_inputs


def is_diverging_layers(
    model_path, aiu_model_handler, cpu_layer_outputs, thresholds, inputs_filename
):
    """Checks if the layer outputs diverge from the CPU run outputs."""
    diverging_layer, status = generate_layerwise_output_diffs(
        aiu_model_handler, inputs_filename, cpu_layer_outputs, thresholds
    )
    divergence = diverging_layer is None and status == SUCCESS
    if divergence:
        print(
            f"DEEPVIEW Threshold test passed for all layers of {model_path}\n"
            "DEEPVIEW========================================================================\n"
        )

    return divergence


def run_layer_io_divergence_mode(model_path, model_type):
    """Runs the layer_io_divergence_mode mode. Uses inputs_filename to get the precaptured inputs if specified by the user.

    Returns True if all layers pass the thresholds test, otherwise False.
    """
    thresholds = get_thresholds(model_path)
    inputs_filename = model_path.split("/")[-1] + ".pkl"

    # Here start the inference on CPU to capture the layer IO
    print("========= Running on CPU to capture layer IO ==========")
    cpu_model_handler = setup_model_handler(
        model_type=model_type,
        model_path=model_path,
        device="cpu",
        prompt="What is the capital of Egypt?",
        insert_forward_hooks=True,
    )
    print("Reached first infer call post compile.....")
    cpu_model_handler.infer()
    print(f"Getting layerwise outputs.....")
    cpu_layer_outputs = get_layerwise_outputs(cpu_model_handler)

    print("========= Running on AIU to capture layer divergence ==========")
    aiu_model_handler = create_model_handler(
        model_type=model_type,
        model_path=model_path,
        device="aiu",
        prompt="What is the capital of Egypt?",
    )

    print("Capturing layerwise inputs....")
    aiu_model_handler.layer_inputs = get_layerwise_inputs(
        model_type, model_path, inputs_filename
    )

    print("Capturing layerwise outputs and calculating divergence....")
    return is_diverging_layers(
        model_path, aiu_model_handler, cpu_layer_outputs, thresholds, inputs_filename
    )
