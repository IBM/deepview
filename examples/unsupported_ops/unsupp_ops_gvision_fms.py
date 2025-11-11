# Command to run script: python3 unsupp_ops_gvision_fms.py > gvision_aiu_out.txt 2>&1
# Standard
import os

# Third Party
from fms.models import get_model
from fms.utils import serialization
from fms.utils.generation import generate, pad_input_ids
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import requests
import torch

# REQUIRED FOR DEEPVIEW AS A LIBRARY
import torch_sendnn

os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

# Local
from deepview.core.unsupported_ops import process_unsupported_ops


def _get_inputs(processor):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\nWhat animal is shown in this image?\n<|assistant|>\n"
    inputs = processor(text=inputs, images=image, return_tensors="pt").to("cpu")
    return inputs


def print_result(result, result_idx: int):
    print(processor.decode(result, skip_special_tokens=True))


if __name__ == "__main__":
    model_path = "ibm-granite/granite-vision-3.2-2b"
    # STEP 1 REQUIRED FOR DEEPVIEW AS A LIBRARY: prep input
    processor = LlavaNextProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    device = "cpu"
    device_type = "aiu"
    max_new_tokens = 20
    min_pad_length = 0
    fixed_prompt_length = 0

    serialization.extend_adapter(
        "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"]
    )

    config_dict = {}
    config_dict["head_dim"] = 128

    # STEP 2 REQUIRED FOR DEEPVIEW AS A LIBRARY: load & Compile (ensure to add compile for sendnn backend)
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float16,
        device_type=device,
        fused_weights=False,
        override_hf_pretrained_config=True,
        text_config=config_dict,
    )

    model.eval()
    model.compile(backend="sendnn")

    # Set Autograd to false
    model.requires_grad_(False)

    # Prepare Inputs
    input_ids = inputs["input_ids"]
    inputs.pop("input_ids")
    input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=4928)
    inputs["mask"] = padding_kwargs["mask"].to(device)
    inputs["position_ids"] = padding_kwargs["position_ids"].to(device)
    input_ids = input_ids.to(device)

    inputs["only_last_token"] = True
    inputs["attn_name"] = "sdpa_causal"

    # STEP 3 REQUIRED FOR DEEPVIEW AS A LIBRARY: call sendnn warmup mode
    with torch_sendnn.warmup_mode():
        # STEP 4 REQUIRED FOR DEEPVIEW AS A LIBRARY: generate call
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
            contiguous_cache=True,
        )

    for i in range(output.shape[0]):
        print_result(output[i], i)

    # STEP 5 REQUIRED FOR DEEPVIEW AS A LIBRARY: Process unsupported Ops
    process_unsupported_ops(True, True)
