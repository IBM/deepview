import torch
import requests
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate, pad_input_ids

# STEP 1 REQUIRED FOR DEEPVIEW AS A LIBRARY
import torch_sendnn
from deepview.core.unsupported_ops import process_unsupported_ops

if __name__ == "__main__":
    model_path = "ibm-ai-platform/Bamba-9B-v2"
    prompt = "What is the capital of Egypt?"

    # STEP 2 REQUIRED FOR DEEPVIEW AS A LIBRARY: prep input
    tokenizer = tokenizers.get_tokenizer(model_path)
    tokens = tokenizer.tokenize(prompt)
    ids_l = tokenizer.convert_tokens_to_ids(tokens)
    if tokenizer.bos_token_id != tokenizer.eos_token_id:
        ids_l = [tokenizer.bos_token_id] + ids_l

    prompt1 = torch.tensor(ids_l, dtype=torch.long, device="cpu")
    input_id, extra_generation_kwargs = pad_input_ids([prompt1], min_pad_length=64)

    # STEP 3 REQUIRED FOR DEEPVIEW AS A LIBRARY: load & Compile (ensure to add compile for sendnn backend)
    model = get_model(
        "hf_pretrained",
        variant=model_path,
        device_type="cpu",
        data_type=torch.float16,
        fused_weights=False,
    )
    model.eval()
    model.compile(backend="sendnn")

    # Set Autograd to false
    model.requires_grad_(False)
    
    # STEP 4 REQUIRED FOR DEEPVIEW AS A LIBRARY: generate call with sendnn warmup
    with torch_sendnn.warmup_mode(skip_compilation=True):
        extra_generation_kwargs["only_last_token"] = True
        eos_token_id = None
        max_len = model.config.max_expected_seq_len
        result = generate(
            model,
            input_id,
            max_new_tokens=2,
            use_cache=True,
            do_sample=False,
            max_seq_len=max_len,
            eos_token_id=eos_token_id,
            contiguous_cache=True,
            extra_kwargs=extra_generation_kwargs,
        )

    # STEP 5 REQUIRED FOR DEEPVIEW AS A LIBRARY: Process unsupported Ops
    process_unsupported_ops(True, False)
    