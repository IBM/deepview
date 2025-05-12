# Allows running specific modules/layers in isolation with preserved context
import torch
import re
import sys
import os
import time
import json
import subprocess
from fms.models import get_model
from fms.utils.generation import generate, pad_input_ids
from fms.utils import tokenizers
from torch_sendnn import torch_sendnn
from torch import distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.generate_minimal_repro import generate_repro_code_unsupported_ops,  generate_repro_code_layer_debugging


class PrintOutput:
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'a')
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return self.stream.isatty()
    def fileno(self):
        return self.stream.fileno()
    def close(self):
        self.file.close()


def set_environment():
    old_output_file = 'model_output.txt'
    if os.path.exists(old_output_file):
        print("Deleting the old model_output.txt..............")
        os.remove(old_output_file)
    os.environ['DTLOG_LEVEL'] = 'error'
    os.environ['TORCH_SENDNN_LOG'] = 'CRITICAL'
    os.environ['DT_DEEPRT_VERBOSE'] = '-1'
    os.environ['PYTHONUNBUFFERED'] = '1'


def load_model_and_create_input(model_type, model_path):
    global model, tokenizer, device, prompt, input_id, extra_generation_kwargs
    device = torch.device("cpu")

    # Load the model
    print("Loading model")
    loading_model_time = time.time()
    if model_type == 'fms':
        torch.set_default_dtype(torch.float16)
        model = get_model("hf_pretrained", variant=model_path, model_path = None, device_type="cpu", data_type=torch.float16, source=None, linear_config={"linear_type": "torch_linear"}, distributed_strategy=None, group=dist.group.WORLD, fused_weights=False, attn_layer_indices=[])
        tokenizer = tokenizers.get_tokenizer(model_path)
        
        # Create the prompt input
        prompt = "What is the capital of India?"
        tokens = tokenizer.tokenize(prompt)
        ids_l = tokenizer.convert_tokens_to_ids(tokens)
        if tokenizer.bos_token_id != tokenizer.eos_token_id:
            ids_l = [tokenizer.bos_token_id] + ids_l
        ids_l = torch.tensor(ids_l, dtype=torch.long, device=device)
        input_id = [ids_l]
        input_id, extra_generation_kwargs = pad_input_ids(input_id, min_pad_length=64)

    elif model_type == 'hf':
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        
        # Create the prompt input
        prompt = "What is the capital of India?"
        input_id = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    


    loading_model_time = time.time() - loading_model_time
    print(f"Loading complete, took {loading_model_time:.3f}s")

    model.eval()
    torch.set_grad_enabled(False)

    # Comment out the lines below to run the whole model.
    if hasattr(model, "base_model"):
        model.base_model.layers = model.base_model.layers[:1]
    elif hasattr(model, "layers"):
        model.layers = model.layers[:1]
    else:
        print("No accessible 'base_model' or 'layers' attribute to slice.")

    # Compile the model
    print("Compiling model")
    compiling_model_time = time.time()
    model.compile(backend="sendnn_decoder", dynamic=False)
    compiling_model_time = time.time() - compiling_model_time
    print(f"Compiling complete, took {compiling_model_time:.3f}s")



def infer(model_type):
    if model_type == 'fms':
        extra_generation_kwargs = None
        max_seq_len = max(len(prompt), model.config.max_expected_seq_len)
        result = generate(model,input_id,use_cache=True,do_sample=False,max_new_tokens=8, max_seq_len=max_seq_len,eos_token_id=None,contiguous_cache=True,extra_kwargs=extra_generation_kwargs)
    elif model_type == 'hf':
        generate_ids = model.generate(input_id)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(result)


def process_output_unsupported_ops(tool_output_file, logfile, generate_repro_code_flag):
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.
    flag = False
    unknown_nodes = []
    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        for line in infile:
            if line.startswith("DEBUG TOOL"):
                flag = True
                outfile.write(line)
                match = re.search(r'DEBUG TOOL Caught error for (.*?):', line)
                if match:
                    node = match.group(1)
                    unknown_nodes.append(node)

        unknown_nodes_str = '\n'.join(sorted(set(unknown_nodes)))
        final_line = (
            "DEBUG TOOL========================================================================\n"
            "DEBUG TOOL Unsupported operations list:\n"
            f"{unknown_nodes_str}"
        )
        print(final_line)
        outfile.write(final_line + "\n")
        if not flag:
            no_unsup_op = (
                "DEBUG TOOL========================================================================\n"
                "DEBUG TOOL \033[1mNo unsupported operations detected.\033[0m\n"
            )
            print(no_unsup_op)
            outfile.write(no_unsup_op)

    # Generate reproduction code for unsupported ops (if any)
    if generate_repro_code_flag:
        if flag:
            print("Generating reproduction code")
            generate_repro_code_unsupported_ops()
        else:
            print("No reproduction code generated as no unsupported operations found.")
    return flag

def process_output_layer_debugging(tool_output_file, logfile, generate_repro_code_flag, model_path):
    # All DEBUG TOOL output lines are extracted and saved in tool_output_file.
    
    flag = False
    with open(logfile, "r") as infile, open(tool_output_file, "w") as outfile:
        debug_lines = [line for line in infile if line.startswith("DEBUG TOOL")]
        outfile.writelines(debug_lines)

    # Parse the tool output for failures
    with open(tool_output_file, "r+") as f:
        lines = f.readlines()
        err_msg = next((line for line in reversed(lines) if "DEBUG TOOL update lazy handle for" in line), None)

        print("======================================================")
        if err_msg:
            layer = err_msg.split("for ")[1].split(", input")[0]
            second_run_str = f"DEBUG TOOL second run for {layer}"
            failed_layer = f"Failed layer is {err_msg.split('for')[1]}" if second_run_str not in ''.join(lines) else "No model layer has failed"
            print(f"{err_msg.strip()}\n{failed_layer}")
            print("======================================================")
            f.write(failed_layer + "\n")

            # Prepare repro code if failure detected
            if failed_layer != "No model layer has failed":
                flag = True
                if generate_repro_code_flag:
                    generate_repro_code_layer_debugging(err_msg, layer, model_path)
        else:
            print("No update lazy handle line found.")
    return flag


def insert_forward_hooks():
    print("Inserting forward hooks.............")
    module_instance_names = {}
    def get_instance_names(module, current_depth=0, name='model'):
        module_instance_names[module] = name
        parent=name

        # if we are dealing with array of layers
        array_layers = all(key.isdigit() for key in module._modules.keys())
        for name, child in module._modules.items():
            if array_layers:
                get_instance_names(child, current_depth + 1, parent+'['+name+']')
            else:
                get_instance_names(child, current_depth + 1, parent+'.'+name)
    get_instance_names(model)
    
    hooks = []
    global layer_list
    layer_list = {}
    def hook_fn(module, input, output):
        module_instance = module_instance_names.get(module, 0)
        if len(input) == 0:
            return
        input_shape_str = f"[{', '.join(map(str, input[0].shape))}]"
        input_type = str(input[0].dtype)
        layer_list[module_instance] = {input_shape_str,input_type}

    for name, layer in model.named_modules():
        hooks.append(layer.register_forward_hook(hook_fn))

    return hooks

def remove_forward_hooks(hooks):
    for hook in hooks:
        hook.remove()

def run_individual_layers(logfile, model_path):
    print("Running each layer individually........")
    command1 = [
        'python3', 'core/test_layers.py', '--model_path', model_path
    ]

    # Show output in terminal as well as save in file
    with open(logfile, "a") as f:
        process = subprocess.Popen(command1,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
             print(line, end='')
             f.write(line)

def run_model(model_type, model_path, tool_output_file, deepview_mode, generate_repro_code_flag, logfile='model_output.txt'):
    if generate_repro_code_flag:
        torch_sendnn.preserve_lazyhandle()

    load_model_and_create_input(model_type, model_path)

    # Prints generated by compile_and_infer are shown in terminal as well as saved in logfile.
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    tee_stdout = PrintOutput(logfile, original_stdout)
    tee_stderr = PrintOutput(logfile, original_stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    print("Reached first infer call post compile.....")
    try:
        if 'layer_debugging' in deepview_mode:
            hooks = insert_forward_hooks()

        infer(model_type)  

        if 'layer_debugging' in deepview_mode:
            remove_forward_hooks(hooks)
            with open("model_list.txt", "w") as file:
                json.dump({k: list(v) for k, v in layer_list.items()}, file)
                file.close()
            run_individual_layers(logfile, model_path)

    except Exception as e:
        print(f"Exception occurred: {e}", file=original_stderr)
    finally:
        tee_stdout.flush()
        tee_stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()
        tee_stderr.close()

    # Process the logfile to create the tool_output_file 
    if 'unsupported_op' in deepview_mode:
        process_output_unsupported_ops(tool_output_file, logfile, generate_repro_code_flag)
    if 'layer_debugging' in deepview_mode:
        process_output_layer_debugging(tool_output_file, logfile, generate_repro_code_flag, model_path)

    # Since both our modes produce outputs before this line, commenting out the code after that -- this can be enabled in future
    # Update lazyhandle after first run of inference
    # print("Updating lazyhandle")
    # torch_sendnn.update_lazyhandle()
