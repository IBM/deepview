import time
import torch
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelHandler:
    def __init__(self, model_type, model_path, prompt):
        self.model_type = model_type
        self.model_path = model_path
        self.prompt = prompt
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.input_id = None
        self.hooks = []
        self.layer_list = {}

    def load_and_compile_model(self):
        print("Loading model")
        start = time.time()
        
        if self.model_type == 'fms':
            self.model = get_model(
                "hf_pretrained", None, model_path=self.model_path,
                device_type="cpu", data_type=torch.float16,
                source=None, distributed_strategy=None, fused_weights=False
            )
        elif self.model_type == 'hf':
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

        print(f"Loading complete, took {time.time() - start:.3f}s")

        self.model.eval()
        torch.set_grad_enabled(False)

        if hasattr(self.model, "base_model"):
            self.model.base_model.layers = self.model.base_model.layers[:1]
        elif hasattr(self.model, "layers"):
            self.model.layers = self.model.layers[:1]
        else:
            print("No accessible 'base_model' or 'layers' attribute to slice.")

        print("Compiling model")
        start = time.time()
        self.model.compile(backend="sendnn_decoder", dynamic=False)
        print(f"Compiling complete, took {time.time() - start:.3f}s")

        return self.model

    def prep_input(self):
        if self.model_type == 'fms':
            self.tokenizer = tokenizers.get_tokenizer(self.model_path)
            tokens = self.tokenizer.tokenize(self.prompt)
            ids_l = self.tokenizer.convert_tokens_to_ids(tokens)
            ids_l = [self.tokenizer.bos_token_id] + ids_l
            self.input_id = torch.tensor(ids_l, dtype=torch.long, device=self.device)
        elif self.model_type == 'hf':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            self.input_id = self.tokenizer(self.prompt, add_special_tokens=False, return_tensors='pt').input_ids

    def infer(self):
        if self.model_type == 'fms':
            extra_generation_kwargs = None
            max_seq_len = max(len(self.prompt), self.model.config.max_expected_seq_len)
            result = generate(
                self.model, self.input_id,
                do_sample=False, max_new_tokens=2,
                max_seq_len=max_seq_len,
                extra_kwargs=extra_generation_kwargs
            )
        elif self.model_type == 'hf':
            generate_ids = self.model.generate(self.input_id)
            result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(result)

    def insert_forward_hooks(self):
        print("Inserting forward hooks.............")
        module_instance_names = {}

        def get_instance_names(module, current_depth=0, name='model'):
            module_instance_names[module] = name
            parent = name
            array_layers = all(key.isdigit() for key in module._modules.keys())
            for subname, child in module._modules.items():
                if array_layers:
                    get_instance_names(child, current_depth + 1, f"{parent}[{subname}]")
                else:
                    get_instance_names(child, current_depth + 1, f"{parent}.{subname}")

        get_instance_names(self.model)

        def hook_fn(module, input, output):
            module_instance = module_instance_names.get(module, "unknown")
            if len(input) == 0:
                return
            input_shape_str = f"[{', '.join(map(str, input[0].shape))}]"
            input_type = str(input[0].dtype)
            self.layer_list[module_instance] = {input_shape_str, input_type}

        for name, layer in self.model.named_modules():
            self.hooks.append(layer.register_forward_hook(hook_fn))

    def remove_forward_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
