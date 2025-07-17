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

#!/usr/bin/env python3
"""
DeepView CLI tool for debugging and analyzing deep learning models.
"""

# Standard
import argparse
import os
import sys

# Local
from deepview.core.model_runner import run_model, set_environment


def main():
    """Main entry point for the deepview CLI."""
    parser = argparse.ArgumentParser(
        description="DeepView: A debugging and analyzing tool for models enablement over IBM Spyre AIUs."
    )

    parser.add_argument(
        "--model_type",
        choices=["fms", "hf"],
        required=True,
        default="fms",
        help="The type of model you want to debug - fms or hf.",
    )

    parser.add_argument(
        "--model", required=True, help="Model name in HF format or model path"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "unsupported_op",
            "layer_debugging",
            "aiu_input_capture",
            "layer_io_divergence",
        ],
        default="unsupported_op",
        help="Modes: [unsupported_op, layer_debugging, aiu_input_capture, layer_io_divergence] (Choose ONLY one). Default is the unsupported_op mode.",
    )

    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Print the stack trace for unsupported ops, valid only with unsupported_op.",
    )

    parser.add_argument(
        "--generate_repro_code",
        action="store_true",
        help="Generate minimal reproducible code for unsupported operation.",
    )

    parser.add_argument(
        "--output_file",
        default="debug_tool_log.txt",
        help="Name of the file in which the debug tool output will be stored.",
    )

    parser.add_argument(
        "--layer_inputs_file",
        default=None,
        help="Name of the file in which AIU layer inputs are stored.",
    )

    args = parser.parse_args()

    if args.layer_inputs_file is not None:
        if args.mode not in ["layer_io_divergence", "aiu_input_capture"]:
            print(
                "Error: --layer_inputs_file is valid only for 'aiu_input_capture' and 'layer_io_divergence' modes."
            )
            sys.exit(1)

    # Setting the environment variables
    set_environment()

    # Run the model
    print("Running the model")
    run_model(
        args.model_type,
        args.model,
        args.output_file,
        args.mode,
        args.show_details,
        args.generate_repro_code,
        args.layer_inputs_file,
    )
    print("DeepView run completed")


if __name__ == "__main__":
    main()
