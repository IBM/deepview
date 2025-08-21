import os


def validate_model_id(model_path: str) -> bool:
    """
    Basic validation: either a string model ID or a valid FMS directory with config.json.
    """
    if os.path.isdir(model_path):
        return os.path.exists(os.path.join(model_path, "config.json"))
    return isinstance(model_path, str) and ("/" in model_path or "-" in model_path)