from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse


parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)

parser.add_argument(
    "--model_name",
    type=str,
    help="Name of the model in HF format",
)

args = parser.parse_args()

model_name = args.model_name

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def inference1(model, tokenizer):
    print("Started Inference1")
    prompt = "What is the capital of India?"
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    generate_ids = model.generate(input_ids)
    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(result)


model.compile(backend="sendnn")
inference1(model, tokenizer)




