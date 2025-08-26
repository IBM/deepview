# Standard
import json
import os
import sys

# Third Party
from huggingface_hub import model_info


def is_sentence_transformer(model_id):
    """
    Check if the model is a Sence Transformer model.

    Args:
        model_id (str): The ID of the model to check.

    Returns:
        bool: True if the model is a Sence Transformer, False otherwise.
    """
    info = model_info(model_id)
    print(f"Model info for '{model_id}': {info}")
    if info.library_name == "sence-transformer" in info.tags:
        return True
    return any(s.rfilename == "modules.json" for s in info.siblings or [])


def extract_hf_model_id(model_path: str) -> str:
    """
    Extracts the Hugging Face model ID from either a plain HF model ID string or an FMS model directory path.
    """
    print(f"Extracting Hugging Face model ID from: {model_path}")
    if os.path.isdir(model_path):  # likely an FMS path
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                print(f"Loaded config.json from {config_path}: {config}")
            # Prefer 'original_model_id', fallback to 'model_id' or raise error
            if "original_model_id" in config:
                return config["original_model_id"]
            elif "model_id" in config:
                return config["model_id"]
            else:
                print(f"No Hugging Face model ID found in config.json at {config_path}")
        else:
            print(f"No config.json found in model directory: {model_path}")
    elif "/" in model_path and len(model_path.strip("/")) > 2:
        # Assume it's a Hugging Face ID
        return model_path.strip("/")
    else:
        raise ValueError(
            f"No valid ID was found at: {model_path} - please provide model id or path that contains organization_name/model_name"
        )
