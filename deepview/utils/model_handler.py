# /*******************************************************************************
#  * Copyright 2025 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# *******************************************************************************/

# Standard
import json
import os
import re
import time

# Third Party
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate, pad_input_ids
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoTokenizer,
)
import torch
import torch_sendnn

MODEL_CLASSES = {
    "auto": AutoModel,
    "sequence_classification": AutoModelForSequenceClassification,
    "question_answering": AutoModelForQuestionAnswering,
    "causal_lm": AutoModelForCausalLM,
    "seq2seq_lm": AutoModelForSeq2SeqLM,
    "image_classification": AutoModelForImageClassification,
    "object_detection": AutoModelForObjectDetection,
    "zero_shot_image_classification": AutoModelForZeroShotImageClassification,
    "vision2seq": AutoModelForVision2Seq,
    "visual_question_answering": AutoModelForVisualQuestionAnswering,
    "sentence": SentenceTransformer,
}


def extract_hf_model_id(model_path: str) -> str:
    """
    Extracts the Hugging Face model ID from either a plain HF model ID string or an FMS model directory path.
    """
    if os.path.isdir(model_path):  # likely an FMS path
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            # Prefer 'original_model_id', fallback to 'model_id' or raise error
            if "original_model_id" in config:
                return config["original_model_id"]
            elif "model_id" in config:
                return config["model_id"]
            else:
                print(f"No Hugging Face model ID found in config.json at {config_path}")
        else:
            print(f"No config.json found in model directory: {model_path}")
    elif "/" in model_path and len(model_path.strip("/")) > 2:
        # Assume it's a Hugging Face ID
        return model_path.strip("/")
    else:
        raise ValueError(
            f"No valid ID was found at: {model_path} - please provide model id or path that contains organization_name/model_name"
        )


def validate_model_id(model_path: str) -> bool:
    """
    Basic validation: either a string model ID or a valid FMS directory with config.json.
    """
    if os.path.isdir(model_path):
        return os.path.exists(os.path.join(model_path, "config.json"))
    return isinstance(model_path, str) and ("/" in model_path or "-" in model_path)


def convert_attr_path(attr_path):
    """Converts the name of the modules to match the format in thresholds file."""
    if attr_path:
        attr_path = "model." + attr_path

        def replace_numeric_attr(match):
            number = match.group(1)
            tail = match.group(2)
            return f"[{number}]{tail}"

        pattern = re.compile(r"\.(\d+)(\.|$)")
        converted = pattern.sub(replace_numeric_attr, attr_path)
    else:
        converted = "model"
    return converted


def setup_model_handler(
    model_type,
    model_path,
    device="aiu",
    prompt="What is the capital of Egypt?",
    safe_warmup=False,
    insert_forward_hooks=False,
):
    handler = ModelHandler(
        model_type=model_type,
        model_path=model_path,
        device=device,
        prompt=prompt,
    )
    handler.load_and_compile_model()
    handler.prep_input()
    if insert_forward_hooks:
        handler.insert_forward_hooks()
    if safe_warmup:
        handler.safe_warmup()

    return handler


class ModelHandler:
    """Handles loading, compiling, input preparation, inference, and debugging using hooks for ML models.

    Supports both custom fms and hf models.
    Automatically infers model class for HF models and loads accordingly.

    Attributes:
        model_type (str): Type of model ("fms" or "hf").
        model_path (str): Path to model checkpoint.
        model_class (str, optional): Specific model class for HF models.
        prompt (str): Text prompt used for input preparation.
        device (torch.device): Device to run the model on (CPU).
        model (torch.nn.Module): Loaded and compiled model instance.
        tokenizer: Tokenizer instance appropriate to the model type.
        input_id: Prepared input tokens/tensors.
        hooks (list): List of forward hooks registered on model layers.
        layer_list (dict): Stores layer input shapes and data types from hooks.
        extra_generation_kwargs (dict, optional): Extra kwargs for generation.
        batch_size (int): Batch size for inputs (default 1).
        min_pad_length (int): Minimum padding length for inputs.
        max_new_tokens (int): Number of tokens to generate during inference.

    Methods:
        _infer_model_class(model_path):
            Infers the Hugging Face model class based on the model's config or files.

        load_and_compile_model():
            Loads the model from path and compiles it with 'sendnn' backend.

        prep_input():
            Prepares tokenized inputs for the model based on model type.

        infer():
            Runs inference on the prepared inputs. Uses generation methods if applicable.

        insert_forward_hooks():
            Inserts forward hooks to capture layer input shapes and types during inference.

        remove_forward_hooks():
            Removes all registered forward hooks from the model.

        get_layer_io():
            Extracts inputs and outputs captured by forward hooks into dictionaries.

        clear_layer_io():
            Clears the captured inputs and outputs from the model's modules.

        safe_warmup():
            Performs a warmup pass on the model without updating lazy handles.

        warmup():
            Performs a warmup pass on the model to initialize it for inference.
    """

    def __init__(self, model_type, model_path, device, prompt, model_class=None):
        """Initialize ModelHandler with model configuration.

        Args:
            model_type (str): Type of the model - hf or fms.
            model_path (str): Path of model checkpoint.
            prompt (str): Prompt text for model inference.
            model_class (str, optional): Specific model class to use. Defaults to None.
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model_class = model_class
        self.prompt = prompt
        self.device = torch.device("cpu")
        self.device_to_run = device
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.input_id = None
        self.hooks = []
        self.layers_ios = {}
        self.extra_generation_kwargs = None
        self.batch_size = 1
        self.min_pad_length = 64
        self.max_new_tokens = 2

    def _infer_model_class(self, model_path):
        """Infer the model class based on the model configuration or repo contents.

        Args:
            model_path (str): Path to model checkpoint.

        Returns:
            str: Inferred model class name such as 'causal_lm', 'sequence_classification', 'sentence', etc.
        """
        # First check if it of type sentence transformer
        try:
            # Third Party
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=model_path, filename="modules.json")
            return "sentence"
        except Exception as e:
            pass

        config = AutoConfig.from_pretrained(model_path)
        arch = config.architectures[0].lower() if config.architectures else ""

        if "sequenceclassification" in arch:
            return "sequence_classification"
        elif "questionanswering" in arch:
            return "question_answering"
        elif "causallm" in arch:
            return "causal_lm"
        elif "seq2seq" in arch:
            return "seq2seq_lm"
        elif "imageclassification" in arch:
            return "image_classification"
        elif "objectdetection" in arch:
            return "object_detection"
        elif "zeroshotimageclassification" in arch:
            return "zero_shot_image_classification"
        elif "vision2seq" in arch:
            return "vision2seq"
        elif "visualquestionanswering" in arch:
            return "visual_question_answering"

        # Fallback to AutoModel
        return "auto"

    def load_and_compile_model(self):
        """Load and compile the model based on the model type and path.

        Returns:
            torch.nn.Module: The loaded and compiled PyTorch model.
        """
        print("Loading model")
        start = time.time()

        if self.model_type == "fms":
            # This get_model call assumes locally downloaded weights
            self.model = get_model(
                "hf_pretrained",
                variant=self.model_path,
                device_type="cpu",
                data_type=torch.float16,
                fused_weights=False,
            )
        elif self.model_type == "hf":
            # TODO: we can do specific handling per model class but for now everything apart from CausalLM is treated as AutoModel
            # Note: SentenceTransformer has to be loaded as AutoModel as torch.compile does not work for SentenceTransformer
            self.model_class = self._infer_model_class(self.model_path)
            print("---------------------------------------------------------")
            print(f"Model class is {self.model_class}")
            print("---------------------------------------------------------")

            model_class = MODEL_CLASSES[self.model_class]
            if self.model_class == "causal_lm":
                self.model = model_class.from_pretrained(self.model_path)
            else:
                self.model = AutoModel.from_pretrained(self.model_path)

        print(f"Loading complete, took {time.time() - start:.3f}s")

        self.model.eval()
        torch.set_grad_enabled(False)

        print("Compiling model")
        start = time.time()
        if self.device_to_run == "aiu":
            self.model.compile(backend="sendnn", dynamic=False)
        elif self.device_to_run == "cpu":
            self.model.compile(backend="inductor")
        else:
            print("Device not supported by Deepview yet.")
        print(f"Compiling complete, took {time.time() - start:.3f}s")

        return self.model

    def prep_input(self):
        """Prepare input tensors and tokenizers based on the model type and prompt."""
        if self.model_type == "fms":
            self.tokenizer = tokenizers.get_tokenizer(self.model_path)
            tokens = self.tokenizer.tokenize(self.prompt)
            ids_l = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.tokenizer.bos_token_id != self.tokenizer.eos_token_id:
                ids_l = [self.tokenizer.bos_token_id] + ids_l

            prompt1 = torch.tensor(ids_l, dtype=torch.long, device=self.device)
            self.input_id, self.extra_generation_kwargs = pad_input_ids(
                [prompt1], min_pad_length=self.min_pad_length
            )
        elif self.model_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.input_id = self.tokenizer(
                [self.prompt], padding=True, truncation=True, return_tensors="pt"
            )

    def _generate_output(self, is_warmup):
        """Calling generate function based on model_type."""
        if self.model_type == "fms":
            self.extra_generation_kwargs["only_last_token"] = True
            if is_warmup:
                eos_token_id = None
                max_len = self.model.config.max_expected_seq_len
            else:
                eos_token_id = self.tokenizer.eos_token_id
                if (
                    hasattr(self.model.config, "ntk_scaling")
                    and self.model.config.ntk_scaling
                ):
                    max_len = max(
                        len(self.prompt), self.model.config.max_expected_seq_len
                    )
                else:
                    max_len = self.model.config.max_expected_seq_len
            result = generate(
                self.model,
                self.input_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
                max_seq_len=max_len,
                eos_token_id=eos_token_id,
                contiguous_cache=True,
                extra_kwargs=self.extra_generation_kwargs,
            )
        elif self.model_type == "hf":
            if self.model_class in ["causal_lm"]:
                input_ids = self.input_id["input_ids"]
                attention_mask = self.input_id.get("attention_mask", None)

                generate_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  ## Somehow taking True as default which is resulting in error for models like Llama
                )
                result = self.tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            else:
                result = self.model(**self.input_id)
        return result

    def safe_warmup(self):
        """Perform warmup on the prepared input based on the model type, but skip update_lazyhandle()."""
        with torch_sendnn.warmup_mode(skip_compilation=True):
            self._generate_output(True)

    def warmup(self):
        """Perform warmup on the prepared input based on the model type."""
        with torch_sendnn.warmup_mode():
            self._generate_output(True)

    def infer(self):
        """Perform inference on the prepared input based on the model type."""
        return self._generate_output(False)

    def insert_forward_hooks(self):
        """Insert forward hooks into the model layers to capture input shapes and types during forward pass."""
        print("Inserting forward hooks.............")

        def hook_fn(module, input, kwarg, output):
            if len(input) == 0 and len(kwarg) == 0:
                return
            module._debug_input = input
            module._debug_kwarg = kwarg
            if self.device_to_run == "cpu":
                module._debug_output = output

        for name, layer in self.model.named_modules():
            self.hooks.append(layer.register_forward_hook(hook_fn, with_kwargs=True))

    def remove_forward_hooks(self):
        """Remove all previously registered forward hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_layer_io(self):
        """Get all inputs captured using forward hook."""
        print("Extracting layer IO ...")
        for module_name, module in self.model.named_modules():
            ## Modifying keys to match the layer names which can be used to run the layers later.
            if not hasattr(module, "_debug_input") and not hasattr(
                module, "_debug_kwarg"
            ):
                continue

            name = convert_attr_path(module_name)
            self.layers_ios[name] = {}
            self.layers_ios[name]["args"] = []
            self.layers_ios[name]["kwargs"] = {}
            self.layers_ios[name]["outputs"] = []

            ## Capturing inputs
            if hasattr(module, "_debug_input"):
                inputs = tuple(
                    v.detach() if isinstance(v, torch.Tensor) else v
                    for v in module._debug_input
                )
                ## For these two layers, the input is generated by appending the captured input at the end of the input prompt
                ## while maintaining the shape. Otherwise, these two layers throw error while running.
                if (self.device_to_run == "aiu") and (
                    (name == "model") or (name == "model.base_model")
                ):
                    inputs = list(inputs)
                    shift_len = inputs[0].shape[-1]
                    shifted_part = self.input_id[:, shift_len:]
                    inputs[0] = torch.cat((shifted_part, inputs[0]), dim=1)
                    ## For base_model layer, input is padded with zeros in the front to make length 64.
                    if name == "model.base_model" and len(inputs) > 1:
                        current_width = inputs[1].shape[-1]
                        pad_len = 64 - current_width
                        padding = inputs[1].new_zeros(1, pad_len)
                        new_tensor = torch.cat((padding, inputs[1]), dim=1)
                        inputs[1] = new_tensor
                    self.layers_ios[name]["args"] = tuple(inputs)
                else:
                    self.layers_ios[name]["args"] = inputs

            ## Capturing kwargs
            if hasattr(module, "_debug_kwarg"):
                self.layers_ios[name]["kwargs"] = module._debug_kwarg

            ## Capturing outputs
            if hasattr(module, "_debug_output"):
                self.layers_ios[name]["outputs"] = module._debug_output
        ## The following lines basically rearrange the keys of the layer inputs dict to place the model and base_model layers in the end, such that the
        ## layers are run before those two. This is done in order to ensure that the offending layer can be captured. Otherwise, if we run model/base_model
        ## first, if there is any offending layer, the whole thing fails without giving any idea of the offending layer.
        if self.model_type == "fms":
            layers = list(self.layers_ios.keys())
            if "model.base_model" in layers:
                first_two_keys = ["model.base_model", "model"]
            elif "model.shared" in layers:
                first_two_keys = ["model.shared", "model"]
            self.layers_ios = {
                k: self.layers_ios[k] for k in layers[2:] + first_two_keys
            }

    def clear_layer_io(self):
        """Clear all inputs/outputs captured using forward hook."""
        print("Clearing layer IO ...")
        for name, module in self.model.named_modules():
            if hasattr(module, "_debug_input"):
                module._debug_input = None
            if hasattr(module, "_debug_kwarg"):
                module._debug_kwarg = None
            if hasattr(module, "_debug_output"):
                module._debug_output = None
