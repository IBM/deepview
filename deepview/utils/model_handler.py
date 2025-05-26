import time
import torch
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate, pad_input_ids

from urllib.request import urlopen
from PIL import Image

from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    AutoProcessor,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForZeroShotImageClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
)
from sentence_transformers import SentenceTransformer

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

class ModelHandler:
    def __init__(self, model_type, model_path, prompt, model_class=None):
        self.model_type = model_type
        self.model_path = model_path
        self.model_class = model_class
        self.prompt = prompt
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.input_id = None
        self.hooks = []
        self.layer_list = {}
        self.extra_generation_kwargs = None
        self.batch_size = 1
        self.min_pad_length = 64
        self.max_new_tokens = 2

    def _infer_model_class(self, model_path):
        #First check if it of type sentence transformer
        try:
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
        elif "vision2seq" in arch or "conditionalgeneration" in arch:
            return "vision2seq"
        elif "visualquestionanswering" in arch:
            return "visual_question_answering"

        #Fallback to AutoModel
        return "auto"
        

    def load_and_compile_model(self):
        print("Loading model")
        start = time.time()
        
        if self.model_type == 'fms':
            # This get_model call assumes locally downloaded weights
            self.model = get_model(
                "hf_pretrained",
                model_path=self.model_path,
                device_type="cpu",
                data_type=torch.float16,
                fused_weights=False
            )
        elif self.model_type == 'hf':
            # TODO: we can do specific handling per model class but for now everything apart from CausalLM is treated as AutoModel
            # Note: SentenceTransformer has to be loaded as AutoModel as torch.compile does not work for SentenceTransformer
            self.model_class = self._infer_model_class(self.model_path)
            print("---------------------------------------------------------")
            print(f"Model class is {self.model_class}")
            print("---------------------------------------------------------")

            model_class = MODEL_CLASSES[self.model_class]
            if self.model_class == "causal_lm":
                self.model = model_class.from_pretrained(self.model_path)
            elif self.model_class == "vision2seq":
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path)
            else:
                self.model = AutoModel.from_pretrained(self.model_path)


        print(f"Loading complete, took {time.time() - start:.3f}s")

        self.model.eval()
        torch.set_grad_enabled(False)

        print("Compiling model")
        start = time.time()
        if not self.model_class == "vision2seq":
            self.model.compile(backend="sendnn_decoder", dynamic=False)
            print(f"Compiling complete, took {time.time() - start:.3f}s")
        else:
            self.model = torch.compile(self.model, backend="sendnn_decoder")
            print(f"Compiling complete, took {time.time() - start:.3f}s")

        return self.model

    def prep_input(self):
        if self.model_type == 'fms':
            self.tokenizer = tokenizers.get_tokenizer(self.model_path)
            tokens = self.tokenizer.tokenize(self.prompt)
            ids_l = self.tokenizer.convert_tokens_to_ids(tokens)
            if self.tokenizer.bos_token_id != self.tokenizer.eos_token_id:
                ids_l = [self.tokenizer.bos_token_id] + ids_l

            prompt1 = torch.tensor(ids_l, dtype=torch.long, device=self.device)
            self.input_id, self.extra_generation_kwargs = pad_input_ids([prompt1], min_pad_length=self.min_pad_length)
        elif self.model_type == 'hf':
            if self.model_class in ['vision2seq']:
                self.processor = AutoProcessor.from_pretrained(self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.input_id = self.tokenizer([self.prompt], padding=True, truncation=True, return_tensors='pt')

    def infer(self):
        if self.model_type == 'fms':
            self.extra_generation_kwargs["only_last_token"] = True
            result = generate(
                self.model,
                self.input_id,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
                max_seq_len=self.model.config.max_expected_seq_len,
                eos_token_id=self.tokenizer.eos_token_id,
                contiguous_cache=True,
                extra_kwargs=self.extra_generation_kwargs,
            )
        elif self.model_type == 'hf':
            if self.model_class in ['vision2seq']:
                messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": "Convert this page to docling."}
                                ]
                            },
                        ]

                image = Image.open(urlopen("https://upload.wikimedia.org/wikipedia/commons/7/76/GazettedeFrance.jpg"))
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

                generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
                prompt_length = inputs.input_ids.shape[1]
                trimmed_generated_ids = generated_ids[:, prompt_length:]
                doctags = self.processor.batch_decode(
                    trimmed_generated_ids,
                    skip_special_tokens=False,
                )[0].lstrip()

                result = doctags
            elif self.model_class in ['causal_lm']:
                input_ids = self.input_id['input_ids']
                attention_mask = self.input_id.get('attention_mask', None)
                generate_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens)
                result = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            else:
                result = self.model(**self.input_id)
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
