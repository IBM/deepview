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

# Standard
import re
import time

# Third Party
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

# Local
from deepview.utils.ModelHandler.HF.hugging_face_utils import is_sentence_transformer

MODEL_CLASSES = {
    "auto": AutoModel,
    "sequenceclassification": AutoModelForSequenceClassification,
    "questionanswering": AutoModelForQuestionAnswering,
    "causallm": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "imageclassification": AutoModelForImageClassification,
    "objectdetection": AutoModelForObjectDetection,
    "zeroshotimageclassification": AutoModelForZeroShotImageClassification,
    "vision2seq": AutoModelForVision2Seq,
    "visualquestionanswering": AutoModelForVisualQuestionAnswering,
    "sentence": SentenceTransformer,
}


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


class ModelHandlerBase:
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
        _get_model_class(model_path):
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
        self.layer_list = {}
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.extra_generation_kwargs = None
        self.batch_size = 1
        self.min_pad_length = 64
        self.max_new_tokens = 2

    def _get_model_class(self, model_path):
        """Infer the model class based on the model configuration or repo contents.

        Args:
            model_path (str): Path to model checkpoint.

        Returns:
            str: Inferred model class name such as 'causal_lm', 'sequence_classification', 'sentence', etc.
        """

        arch = self._get_model_architecture(model_path)
        model_class = MODEL_CLASSES.get(arch, None)

        if not model_class:
            return MODEL_CLASSES.get("auto", AutoModel)

        print("---------------------------------------------------------")
        print(f"Model class is {arch}")
        print("---------------------------------------------------------")
        return model_class

    def _get_model_architecture(self, model_path):
        """Get the architecture of the model from its configuration.

        Args:
            model_path (str): Path to the model checkpoint.
        Returns:
            str: The architecture type of the model, such as 'causallm', 'sequenceclassification', etc.
        """

        config = AutoConfig.from_pretrained(model_path)
        return config.architectures[0].lower() if config.architectures else ""

    def _load_model(self):
        NotImplementedError(
            "This method should be implemented in subclasses to load the model."
        )

    def load_model(self):
        """Load the model based on the model type and path.

        Raises:
            ValueError: If the model type is not supported.
        """
        print("Loading model")
        start = time.time()
        self._load_model()
        print(f"Loading complete, took {time.time() - start:.3f}s")

    def compile_model(self):
        """Compile the model for the specified device using the appropriate backend.
        Raises:
            ValueError: If the device is not supported.
        """
        compile_model = {
            "aiu": lambda: self.model.compile(backend="sendnn", dynamic=False),
            "cpu": lambda: self.model.compile(backend="inductor"),
        }

        if self.device_to_run not in compile_model:
            print("Device not supported by Deepview yet.")
            raise ValueError(f"Unsupported device: {self.device_to_run}")

        print("Compiling model")
        start = time.time()
        compile_model[self.device_to_run]()
        print(f"Compiling complete, took {time.time() - start:.3f}s")

    def load_and_compile_model(self):
        """Load and compile the model based on the model type and path.

        Returns:
            torch.nn.Module: The loaded and compiled PyTorch model.
        """

        self.load_model()
        self.model.eval()
        torch.set_grad_enabled(False)
        self.compile_model()

        return self.model

    def _prep_input(self):
        """Prepare input tensors for the model based on the model type."""
        NotImplementedError(
            "This method should be implemented in subclasses to prepare the input tensors."
        )

    def prep_input(self):
        """Prepare input tensors and tokenizers based on the model type and prompt."""
        print("Preparing input")
        start = time.time()
        self._prep_input()
        print(f"Input preparation complete, took {time.time() - start:.3f}s")

    def _generate_output(self, is_warmup):
        """Calling generate function based on model_type."""
        NotImplementedError(
            "This method should be implemented in subclasses to generate output."
        )

    def warmup(self, safe=True):
        """Perform warmup on the prepared input based on the model type."""
        with torch_sendnn.warmup_mode(skip_compilation=safe):
            self._generate_output(True)

    def infer(self):
        """Perform inference on the prepared input based on the model type."""
        return self._generate_output(False)

    def insert_forward_hooks(self):
        """Insert forward hooks into the model layers to capture input shapes and types during forward pass."""
        print("Inserting forward hooks.............")

        def hook_fn(module, input, output):
            if len(input) == 0:
                return
            module._debug_input = input
            if self.device_to_run == "cpu":
                module._debug_output = output

        for name, layer in self.model.named_modules():
            self.hooks.append(layer.register_forward_hook(hook_fn))

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
            name = convert_attr_path(module_name)
            ## Capturing inputs
            if hasattr(module, "_debug_input"):
                inputs = tuple(
                    v.detach()
                    for v in module._debug_input
                    if isinstance(v, torch.Tensor)
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

                    self.layer_inputs[name] = tuple(inputs)
                else:
                    self.layer_inputs[name] = inputs
            ## Capturing outputs
            if hasattr(module, "_debug_output"):
                self.layer_outputs[name] = module._debug_output
        ## The following lines basically rearrange the keys of the layer inputs dict to place the model and base_model layers in the end, such that the
        ## layers are run before those two. This is done in order to ensure that the offending layer can be captured. Otherwise, if we run model/base_model
        ## first, if there is any offending layer, the whole thing fails without giving any idea of the offending layer.
        if self.model_type == "fms":
            layers = list(self.layer_inputs.keys())
            if "model.base_model" in layers:
                first_two_keys = ["model.base_model", "model"]
            elif "model.shared" in layers:
                first_two_keys = ["model.shared", "model"]
            self.layer_inputs = {
                k: self.layer_inputs[k] for k in layers[2:] + first_two_keys
            }

    def clear_layer_io(self):
        """Clear all inputs/outputs captured using forward hook."""
        print("Clearing layer IO ...")
        for name, module in self.model.named_modules():
            if hasattr(module, "_debug_input"):
                module._debug_input = None
            if hasattr(module, "_debug_output"):
                module._debug_output = None
