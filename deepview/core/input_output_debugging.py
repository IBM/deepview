# Standard
from pathlib import Path
import subprocess

# Local
from deepview.core.individual_layer_run_with_inputs import run_layers


def generate_individual_layer_output(model_path, model_type, layer_list, layer_inputs):
    """Generates layer outputs by running each layer of the model individually on the inputs collected in forward pass.

    Iterates over the all layers and captures the outputs of the layers
    At the end of the run, stores the inputs and outputs per layer in a pkl file.

    Args:
        model_path (str): Path to the model checkpoint directory.
        model_type (str): Model type, either 'hf' (HuggingFace) or 'fms' (Foundation Model Stack).
        layer_list (dict): Dictionary mapping layer/module names to a set containing input shape and data type.
    """
    print(layer_inputs)
    print("Running each layer individually........")
    layers_done = []
    failed_layer = "No failed layer"
    for str_layer, val in layer_list.items():
        sub_layer = (
            str_layer.rsplit(".", str_layer.count(".") - 3)[0]
            if str_layer.count(".") > 3
            else str_layer
        )
        val_list = list(val)
        if sub_layer in layers_done:
            continue

        print(
            "DEEPVIEW========================================================================\n"
            f"DEEPVIEW Running {sub_layer}"
        )

        # layer_run = run_layers(model_path, sub_layer, input)
        # command1 = ["python3", "-c", layer_run]
        # process = subprocess.run(
        #     command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        # )

        # for line in process.stdout:
        #     print(line, end="")
        # if process.returncode != 0:
        #     print(
        #         "DEEPVIEW========================================================================\n"
        #         f"DEEPVIEW \033[1mError running {sub_layer}, {input_shape}, {datatype}\n\033[0m"
        #         "DEEPVIEW========================================================================\n"
        #     )
        #     failed_layer = sub_layer
        #     break
        # else:
        #     print(
        #         f"DEEPVIEW Successfully ran {sub_layer}, {input_shape}, {datatype}\n"
        #         "DEEPVIEW========================================================================\n"
        #     )

        # layers_done.append(sub_layer)


