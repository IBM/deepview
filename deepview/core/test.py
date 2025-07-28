# Standard
import inspect
import itertools
import os
import pickle

# Third Party
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate, pad_input_ids
from torch import tensor
import torch
import torch_sendnn

os.environ["COMPILATION_MODE"] = "offline_decoder"

model = get_model(
    "hf_pretrained",
    variant="ibm-granite/granite-3.2-8b-instruct",
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

layer = model
target_layer = layer
forward_signature = inspect.signature(target_layer.forward)
expected_args = list(forward_signature.parameters.keys())
print(expected_args)

with open("granite-3.2-8b-instruct.pkl", "rb") as f:
    layer_inputs_dict = pickle.load(f)

inputval = layer_inputs_dict[""]
inputvals = list(inputval)
if len(inputval) < len(expected_args):
    zipped_inputs = list(itertools.zip_longest(expected_args, inputval, fillvalue=None))
else:
    zipped_inputs = list(zip(expected_args, inputval))
kwargs = dict(zipped_inputs)

print(kwargs)

prompt = "What is the capital of Egypt?"
min_pad_length = 64
tokenizer = tokenizers.get_tokenizer("ibm-granite/granite-3.2-8b-instruct")
tokens = tokenizer.tokenize(prompt)
ids_l = tokenizer.convert_tokens_to_ids(tokens)
if tokenizer.bos_token_id != tokenizer.eos_token_id:
    ids_l = [tokenizer.bos_token_id] + ids_l

prompt1 = torch.tensor(ids_l, dtype=torch.long, device=device)
input_id, extra_generation_kwargs = pad_input_ids(
    [prompt1], min_pad_length=min_pad_length
)

print(input_id)
print("-----------------------")
new_x = kwargs["x"].expand(1, 64)
new_new_x = tensor(
    [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            8197,
            438,
            322,
            18926,
            432,
            516,
            4679,
            385,
            49,
            203,
        ]
    ]
)

print(new_new_x)

layer.compile(backend="sendnn", dynamic=False)
with torch_sendnn.warmup_mode():
    result = layer(new_new_x)
result = layer(new_new_x)
print(result)

output_filename = "dv_layer_io_debugging_tmp/model_output_kwargs.pkl"
with open(output_filename, "wb") as f:
    pickle.dump(result, f)
