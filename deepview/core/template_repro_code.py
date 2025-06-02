# Third Party
from fms.models import get_model
from torch_sendnn import torch_sendnn
import torch

model = get_model(
    "hf_pretrained",
    None,
    model_path=modelpath,
    device_type="cpu",
    data_type=torch.float16,
    source=None,
    distributed_strategy=None,
    linear_config={"linear_type": "torch_linear"},
    fused_weights=False,
)
device = torch.device("cpu")
model.eval()
torch.set_grad_enabled(False)
rand_tensor = torch.rand(tuple(input_shape))
data_type = datatype
layer = sub_layer
layer.compile(backend="sendnn_decoder", dynamic=False)
layer(rand_tensor.to(data_type))
torch_sendnn.update_lazyhandle()
layer(rand_tensor.to(data_type))
