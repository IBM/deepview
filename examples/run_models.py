#!/usr/bin/env python3
import subprocess
import argparse

def run_models(mode: str, models: list[str], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for model in models:
            # Build full command string
            full_command = f"deepview --model_type fms --model {model} --mode {mode}"
            f.write(f"=== Running: {full_command} ===\n")
            print(f"Running command for model: {model}")

            try:
                # Run command and capture output (both stdout and stderr)
                result = subprocess.run(
                    full_command,
                    shell=True,
                    text=True,
                    capture_output=True
                )

                # Write output and error to file
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n[stderr]\n" + result.stderr)
                f.write("\n\n")

            except Exception as e:
                f.write(f"[ERROR] Failed to run command for {model}: {e}\n\n")

    print(f"All outputs saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deepview for multiple models and save output to file."
    )
    parser.add_argument(
        "--mode", required=True,
        help="Deepview mode to execute the models"
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="List of model names to run sequentially."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path for saving command results."
    )
    args = parser.parse_args()

    run_models(args.mode, args.models, args.output)
