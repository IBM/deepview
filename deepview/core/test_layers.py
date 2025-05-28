import torch
from fms.models import get_model
from torch_sendnn import torch_sendnn
from fms.utils import serialization, tokenizers

from fms.models.hf.utils import AutoConfig
import pytest
from fms.utils.generation import pad_input_ids

from aiu_fms_testing_utils.testing.validation import (
    extract_validation_information,
    LogitsExtractorHook,
    GoldenTokenHook,
    capture_level_1_metrics,
    filter_failed_level_1_cases,
    validate_level_0,
    top_k_loss_calculator,
)
from aiu_fms_testing_utils.utils import (
    sample_sharegpt_requests,
    ids_for_prompt,
)

from aiu_fms_testing_utils.utils.aiu_setup import dprint

import sys
import json
import argparse
import os 
# import itertools

from deepview.utils.model_handler import ModelHandler

import os


# for validation level 1, the default is a failure rate of 1%
# set this environment variable if you would like to relax that threshold
failure_rate_threshold = 0.01
default_metrics_threshold = (3.0, .001)

# pass custom failure rate threshold as float
if isinstance(failure_rate_threshold, str):
    failure_rate_threshold = float(failure_rate_threshold)

# pass custom default metrics threshold as a comma separated str of floats <cross-entropy threshold>,<mean diff threshold>
if isinstance(default_metrics_threshold, str):
    default_metrics_threshold = tuple([float(m) for m in default_metrics_threshold.split(",")])


def __prepare_inputs(batch_size, seq_length, tokenizer, seed=0):
    prompts_and_sizes = sample_sharegpt_requests(
        None,
        batch_size,
        tokenizer,
        int(seq_length / 2),
        seq_length,
        seed,
    )
    prompt_list = []
    for prompt, _ in prompts_and_sizes:
        prompt_list.append(ids_for_prompt(prompt, tokenizer))

    input_ids, padding_kwargs = pad_input_ids(prompt_list, min_pad_length=seq_length)
    return input_ids, padding_kwargs

def __find_eos_index(reference_tokens, eos_token_id, seq_length, max_new_tokens):
    result = []
    for sentence in reference_tokens:
        found_eos = False
        for token_idx, token in enumerate(sentence[seq_length:]):
            if token.item() == eos_token_id:
                found_eos = True
                result.append(token_idx)
                break
        if not found_eos:
            result.append(max_new_tokens)
    return result


def __filter_before_eos(l, filter_indexes):
    from itertools import groupby

    filtered_results = [
        list(g)[: filter_indexes[k]] for k, g in groupby(l, key=lambda x: x[0])
    ]
    return [item for sublist in filtered_results for item in sublist]


def test_common_shapes(model_path, layer_path, batch_size, seq_length, max_new_tokens):
    torch.manual_seed(42)
    os.environ["COMPILATION_MODE"] = "offline_decoder"

    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/tmp/models/hf_cache"

    dprint(
        f"testing model={model_path}, batch_size={batch_size}, seq_length={seq_length}, max_new_tokens={max_new_tokens}"
    )


    micro_model_kwargs = {"architecture": "hf_pretrained"}
    model_path_kwargs = {"variant": layer_path}

    get_model_kwargs = {
        **model_path_kwargs,
        **micro_model_kwargs,
    }

    tokenizer = tokenizers.get_tokenizer(model_path)

    # prepare the AIU model

    print(layer_path)

    # model = get_model(
    #     device_type="cpu",
    #     data_type=torch.float16,
    #     fused_weights=False,
    #     **get_model_kwargs,
    # )

    model = layer_path

    model.eval()
    torch.set_grad_enabled(False)
    model.compile(backend="sendnn_decoder", dynamic=False)

    # prepare the cpu model
    validation_model = get_model(
        device_type="cpu",
        data_type=torch.float32,
        fused_weights=False,
        **get_model_kwargs,
    )

    # prepare input_ids
    input_ids, padding_kwargs = __prepare_inputs(batch_size, seq_length, tokenizer)

    print("-------------------------------------------------------------------------------------------")
    print(f"DEBUG TOOL first run for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    layer(rand_tensor.to(data_type))

    print(f"DEBUG TOOL update lazy handle for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    torch_sendnn.update_lazyhandle()

    print(f"DEBUG TOOL second run for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    layer(rand_tensor.to(data_type))
    print("-------------------------------------------------------------------------------------------")

    # generate cpu validation info
    cpu_validation_info = extract_validation_information(
        validation_model,
        input_ids,
        max_new_tokens,
        LogitsExtractorHook(),
        attn_algorithm="math",
        **padding_kwargs,
    )

    cpu_static_tokens = cpu_validation_info.get_info("tokens")
    eos_indexes = __find_eos_index(
        cpu_static_tokens, tokenizer.eos_token_id, seq_length, max_new_tokens
    )
    dprint(
        "cpu validation info extracted for validation level 0 and validation level 1 (iter=0)"
    )

    # metric calculator based on the cross-entropy and mean diff for each decode step
    def _metric_calculator(r: torch.Tensor, t: torch.Tensor):
        cross_entropy = torch.nn.CrossEntropyLoss()(
            r, t.softmax(dim=1).to(dtype=torch.float32)
        )
        diff = torch.mean(torch.abs(
            r.softmax(dim=1).to(dtype=torch.float32)
            - t.softmax(dim=1).to(dtype=torch.float32)
        ))
        return (cross_entropy, diff)

    iters = 1024 // max_new_tokens
    ce_fail_responses_list = []
    diff_fail_responses_list = []
    total_tokens = 0
    for i in range(iters):
        # for iteration 0, we have computed the cpu validation info in the prior step for seed=0, so skip
        if i != 0:
            input_ids, padding_kwargs = __prepare_inputs(
                batch_size, seq_length, tokenizer, seed=i
            )
            cpu_validation_info = extract_validation_information(
                    validation_model,
                    input_ids,
                    max_new_tokens,
                    LogitsExtractorHook(),
                    attn_algorithm="math",
                    **padding_kwargs,
                )
            dprint(
                f"cpu validation info extracted for validation level 1 - iter={i}"
            )
            cpu_static_tokens = cpu_validation_info.get_info("tokens")
            eos_indexes = __find_eos_index(
                cpu_static_tokens,
                tokenizer.eos_token_id,
                seq_length,
                max_new_tokens,
            )

        # generate aiu validation info
        aiu_validation_info = extract_validation_information(
            model,
            input_ids,
            max_new_tokens,
            GoldenTokenHook(cpu_static_tokens),
            only_last_token=True,
            **padding_kwargs,
        )
        dprint(f"aiu validation info extracted for validation level 1 - iter={i}")

        # capture all level 1 metrics
        level_1_metrics = capture_level_1_metrics(
            cpu_validation_info.get_info("logits"),
            aiu_validation_info.get_info("logits"),
            top_k_loss_calculator(20, _metric_calculator),
        )
        # only consider those metrics captured prior to the eos
        level_1_metrics = __filter_before_eos(level_1_metrics, eos_indexes)

        ce_threshold, diff_threshold = default_metrics_threshold

        # get all failed responses for each metric
        ce_fail_responses = filter_failed_level_1_cases(
            level_1_metrics, lambda m: m[0] >= ce_threshold
        )
        diff_fail_responses = filter_failed_level_1_cases(
            level_1_metrics,
            lambda m: m[1] >= diff_threshold,
        )

        ce_fail_responses_list.extend(ce_fail_responses)
        diff_fail_responses_list.extend(diff_fail_responses)
        total_tokens += len(level_1_metrics)

    # test the failure rates for across all tokens
    diff_failure_rate = len(diff_fail_responses_list) / total_tokens
    ce_failure_rate = len(ce_fail_responses_list) / total_tokens
    dprint(f"mean diff failure rate: {diff_failure_rate}")
    dprint(f"cross entropy loss failure rate: {ce_failure_rate}")
    assert diff_failure_rate < failure_rate_threshold, (
        f"failure rate for mean diff was too high: {diff_failure_rate}"
    )
    assert ce_failure_rate < failure_rate_threshold, (
        f"failure rate for cross entropy loss was too high: {ce_failure_rate}"
    )


parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_type",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
args = parser.parse_args()

try:
    with open("model_list.txt", 'r') as file:
        layer_list = json.load(file)
except FileNotFoundError:
    print("File 'model_list.txt' not found")
    sys.exit(1)
except json.JSONDecodeError:
    print("Invalid JSON format")
    sys.exit(1)

model_handler = ModelHandler(model_type=args.model_type, model_path=args.model_path, prompt='What is the capital of India?')
model = model_handler.load_and_compile_model()

layers_done = []
# Run inference for each layer
for str_layer, val in layer_list.items():
    sub_layer = str_layer.rsplit('.', str_layer.count('.') - 3)[0] if str_layer.count('.') > 3 else str_layer

    if sub_layer in layers_done:
        continue

    # Determine input shape and data type
    dtype_str, input_shape_str = val if "torch" in val[0] else val[::-1]

    input_shape = eval(input_shape_str)
    data_type = eval(dtype_str)
    rand_tensor = torch.rand(tuple(input_shape))

    # Compile and run
    layer = eval(sub_layer)
    # layer.compile(backend="sendnn_decoder", dynamic=False)

    test_common_shapes(args.model_path, layer, 1, input_shape_str, 128)

    # print("-------------------------------------------------------------------------------------------")
    # print(f"DEBUG TOOL first run for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    # layer(rand_tensor.to(data_type))

    # print(f"DEBUG TOOL update lazy handle for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    # torch_sendnn.update_lazyhandle()

    # print(f"DEBUG TOOL second run for {sub_layer}, input shape {input_shape_str}, data type {dtype_str}")
    # layer(rand_tensor.to(data_type))
    # print("-------------------------------------------------------------------------------------------")

    layers_done.append(sub_layer)


