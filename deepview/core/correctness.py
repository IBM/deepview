# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

import subprocess

# Local
from deepview.core.individual_layer_run import run_layers


def run_individual_layers_output(model_path, model_type, layer_list):
    """Runs each layer of the model individually (in layer debugging mode).

    Args:
        logfile (str): Path to the complete model output log.
        model_path (str): Path to the model checkpoint.
        model_type (str): Type of model hf or fms.
    """
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
        datatype, input_shape = val_list if "torch" in val_list[0] else val_list[::-1]
        if sub_layer in layers_done:
            continue
        # Show output in terminal as well as save in file
        with open(logfile, "a") as f:
            start_line = (
                "DEEPVIEW========================================================================\n"
                f"DEEPVIEW Running {sub_layer}, {input_shape}, {datatype}"
            )
            print(start_line)
            layer_run = run_layers(model_path, sub_layer, input_shape, datatype)
            command1 = ["python3", "-c", layer_run]
            process = subprocess.run(
                command1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")
            if process.returncode != 0:
                error_line = (
                    "DEEPVIEW========================================================================\n"
                    f"DEEPVIEW Error running {sub_layer}, {input_shape}, {datatype}\n"
                    "DEEPVIEW========================================================================\n"
                )
                failed_layer = sub_layer
                print(error_line)
                break
            else:
                success_line = (
                    f"DEEPVIEW Successfully ran {sub_layer}, {input_shape}, {datatype}\n"
                    "DEEPVIEW========================================================================\n"
                )
                print(success_line)
        layers_done.append(sub_layer)
    return failed_layer, input_shape, datatype