# Command to run script: python3 gvision_inference_library.py > gvision_aiu_out.txt 2>&1

from transformers import LlavaNextProcessor
import torch
import requests
from transformers import LlavaNextForConditionalGeneration

#REQUIRED FOR DEEPVIEW AS A LIBRARY
import torch_sendnn
from deepview.core.unsupported_ops import process_unsupported_ops

def _get_inputs(processor):
    from PIL import Image

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\nWhat animal is shown in this image?\n<|assistant|>\n"
    inputs = processor(text=inputs, images=image, return_tensors="pt").to("cpu")
    return inputs


if __name__ == "__main__":
    model_path = "ibm-granite/granite-vision-3.2-2b"
    # STEP 1 REQUIRED FOR DEEPVIEW AS A LIBRARY: prep input
    processor = LlavaNextProcessor.from_pretrained(model_path)
    inputs = _get_inputs(processor)

    # STEP 2 REQUIRED FOR DEEPVIEW AS A LIBRARY: load & Compile (ensure to add compile for sendnn backend)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path).to("cpu")
    model.eval()
    model.compile(backend="sendnn")
    
    # Set Autograd to false
    #model.requires_grad_(False)

    # STEP 3 REQUIRED FOR DEEPVIEW AS A LIBRARY: call sendnn warmup mode
    with torch_sendnn.warmup_mode(skip_compilation=True):
        # STEP 4 REQUIRED FOR DEEPVIEW AS A LIBRARY: generate call
        output = model.generate(
            **inputs, max_new_tokens=20, use_cache=True, do_sample=False
        )
    print(processor.decode(output[0], skip_special_tokens=True))

    # STEP 4 REQUIRED FOR DEEPVIEW AS A LIBRARY: Process unsupported Ops
    process_unsupported_ops(True, True) 