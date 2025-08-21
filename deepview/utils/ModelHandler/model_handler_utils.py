# Local
from deepview.utils.ModelHandler.model_handler_fms import ModelHandlerFMS
from deepview.utils.ModelHandler.model_handler_hf import ModelHandlerHF


def create_model_handler(model_type, model_path, device, prompt):
    """Factory function to create the appropriate ModelHandler based on model type."""
    ModelHandler = {
        "fms": ModelHandlerFMS,
        "hf": ModelHandlerHF,
    }.get(model_type)
    if ModelHandler is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device=device,
        prompt=prompt,
    )
    return handler


def setup_model_handler(
    model_type,
    model_path,
    device="aiu",
    prompt="What is the capital of Egypt?",
    safe_warmup=True,
    insert_forward_hooks=False,
):
    handler = create_model_handler(model_type, model_path, device, prompt)
    handler.load_and_compile_model()
    handler.prep_input()
    if insert_forward_hooks:
        handler.insert_forward_hooks()
    if device == "aiu":
        handler.warmup(safe=safe_warmup)

    return handler
