def run_layers(modelpath, sub_layer, input_shape, datatype):
    """Generates a minimal Python script to reproduce a layer-level failure in layer debugging mode.

    The generated code loads the model, compiles the specified sub-layer using the `sendnn` backend,
    and runs inference twice to simulate lazy compilation and execution.

    Args:
        modelpath (str): Path to the model checkpoint.
        sub_layer (str): The sub-layer (module) name to compile and test.
        input_shape (str): Input tensor shape.
        datatype (str): Torch datatype for the input tensor (e.g., 'torch.float16').

    Returns:
        str: A complete Python script as a string that can be saved and executed to reproduce the failure.
    """
    return f"""
from fms.models import get_model
import torch_sendnn
import torch
import os
os.environ["COMPILATION_MODE"] = "offline_decoder"
model = get_model(
    "hf_pretrained",
    None,
    model_path='{modelpath}',
    device_type="cpu",
    data_type=torch.float16,
    source=None,
    distributed_strategy=None,
    linear_config={{"linear_type": "torch_linear"}},
    fused_weights=False,
)
device = torch.device("cpu")
model.eval()
torch.set_grad_enabled(False)
rand_tensor = torch.rand(tuple({input_shape}))
data_type = {datatype}
layer = {sub_layer}
layer.compile(backend="sendnn", dynamic=False)
print(f"Warmup of layer {sub_layer}, input shape {input_shape}, data type {datatype}")
with torch_sendnn.warmup_mode():
   layer(rand_tensor.to(data_type))
print(f"Second run of the layer {sub_layer}, input shape {input_shape}, data type {datatype}")
layer(rand_tensor.to(data_type))
"""
