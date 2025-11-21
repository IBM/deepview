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

# STEP 1 REQUIRED FOR DEEPVIEW AS A LIBRARY
import torch_sendnn

# Local
from deepview.core.unsupported_ops import process_unsupported_ops

# Set Required Environment Variable
os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

def _get_inputs(processor):
    device = torch.device("cpu")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\nWhat animal is shown in this image?\n<|assistant|>\n"
    inputs = processor(text=inputs, images=image, return_tensors="pt").to(device)
    return inputs


def _setup_dynamo_cache(max_new_tokens, min_pad_length, fixed_prompt_length):
    _target_cache_size = max(
        int(max_new_tokens * 2),
        int(min_pad_length * 2.5),
        int(fixed_prompt_length * 2.5),
    )
    _prompt_size = max(int(min_pad_length), int(fixed_prompt_length))
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        if _target_cache_size > torch._dynamo.config.accumulated_cache_size_limit:
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = _target_cache_size
            print(
                f"NOTICE: Adjusting torch._dynamo.config.accumulated_cache_size_limit from {_prev} to {torch._dynamo.config.accumulated_cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {max_new_tokens}"
            )

    if _target_cache_size > torch._dynamo.config.cache_size_limit:
        _prev = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = _target_cache_size
        print(
            f"NOTICE: Adjusting torch._dynamo.config.cache_size_limit from {_prev} to {torch._dynamo.config.cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {max_new_tokens}"
        )


def infer(model, input_ids, inputs, max_new_tokens, linear_config):
    with torch_sendnn.warmup_mode():
        output = generate(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            max_seq_len=input_ids.shape[1] + max_new_tokens,
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
            timing="per-token",
            contiguous_cache=True,
        )
    return output


def print_result(result, result_idx: int):
    print(processor.decode(result, skip_special_tokens=True))


if __name__ == "__main__":
    # Set up dynamo cache
    max_new_tokens = 4
    min_pad_length = 4992
    fixed_prompt_length = 0
    
    _setup_dynamo_cache(max_new_tokens, min_pad_length, fixed_prompt_length)
    
    model_path = "ibm-granite/granite-vision-3.3-2b"
    device = torch.device("cpu")
    device_type = "aiu"
    linear_config = {"linear_type": "torch_linear"}

    # STEP 2 REQUIRED FOR DEEPVIEW AS A LIBRARY: prep input
    processor = LlavaNextProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    input_ids = inputs["input_ids"]
    inputs.pop("input_ids")
    input_ids, padding_kwargs = pad_input_ids(input_ids, min_pad_length=min_pad_length)
    inputs["mask"] = padding_kwargs["mask"].to(device)
    inputs["position_ids"] = padding_kwargs["position_ids"].to(device)
    input_ids = input_ids.to(device)

    inputs["only_last_token"] = True
    inputs["attn_name"] = "sdpa_causal"
    
    # head_dim expansion required for granite vision
    serialization.extend_adapter(
        "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"]
    )
    config_dict = {}
    config_dict["head_dim"] = 128

    # STEP 3 REQUIRED FOR DEEPVIEW AS A LIBRARY: load & Compile (ensure to add compile for sendnn backend)
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type="cpu",
        linear_config=linear_config,
        fused_weights=False,
        override_hf_pretrained_config=True,
        text_config=config_dict,
    )

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn")
    
    # Warmup AIU
    infer(model, input_ids, inputs, max_new_tokens, linear_config)
    
    #STEP 4 REQUIRED FOR DEEPVIEW AS A LIBRARY: call generate
    output, timings = infer(model, input_ids, inputs, max_new_tokens, linear_config)

    if len(output.shape) == 1:
        output = output.unsqueeze(0)

    for i in range(output.shape[0]):
        print_result(output[i], i)

    # STEP 5 REQUIRED FOR DEEPVIEW AS A LIBRARY: Process unsupported Ops
    process_unsupported_ops(True, True)
