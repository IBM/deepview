#!/usr/bin/env python3
# Standard
from datetime import datetime
import argparse
import subprocess

TEST_MODELS: list[str] = [
    "ibm-granite/granite-3.2-2b-instruct",
    "ibm-granite/granite-3.2-8b-instruct",
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-ai-platform/Bamba-9B-v2",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def deepview_model_runner(mode: str, models: list[str], output_file: str, silent: bool):
    def emit(msg: str, end: str = "\n"):
        if not silent:
            print(msg, end=end)

    with open(output_file, "w", encoding="utf-8") as f:
        for model in models:
            full_command = f"deepview --model_type fms --model {model} --mode {mode}"
            header = f"\n=== [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running: {full_command} ===\n"
            emit(header.strip())
            f.write(header)

            try:
                # Stream stdout+stderr live (stderr merged into stdout)
                process = subprocess.Popen(
                    full_command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # line-buffered
                )

                # Stream each line to terminal (unless silent) and to file
                assert process.stdout is not None
                for line in process.stdout:
                    if not silent:
                        print(line, end="")
                    f.write(line)

                process.wait()

                if process.returncode != 0:
                    err_msg = f"[WARNING] Deepview run for {model} {mode} mode exited with code {process.returncode}\n"
                    print(err_msg)
                    f.write(err_msg)

            except Exception as e:
                err_msg = f"[ERROR] Exception while running Deepview {mode} mode for {model}: {e}\n"
                print(err_msg)
                f.write(err_msg)

            f.write(f"=== Finished {mode} for model: {model} ===\n\n")
            f.flush()  # make sure content is written to file immediately

    print(f"\n✅ All runs completed. Output saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Deepview sequentially for multiple models and save results to an output file."
    )
    parser.add_argument(
        "--mode",
        required=True,
        help="Deepview mode to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of model names to run sequentially.",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Run Deepview on the curated set of models defined in this script.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output file path for saving all run logs.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="If set, suppress live terminal output (still logs to file).",
    )
    args = parser.parse_args()

    if args.all_models and args.models:
        parser.error("Use either --models or --all_models, not both.")

    if not args.all_models and not args.models:
        parser.error("Provide --models or pass --all_models to use the curated list.")

    selected_models = TEST_MODELS if args.all_models else args.models

    deepview_model_runner(args.mode, selected_models, args.output_file, args.silent)
