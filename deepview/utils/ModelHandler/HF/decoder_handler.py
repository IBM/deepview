# Third Party
from transformers import AutoTokenizer

# Local
from deepview.utils.ModelHandler.model_handler_base import ModelHandlerBase


class DecoderHandlerHF(ModelHandlerBase):
    """Handles Hugging Face models with specific input preparation and compilation methods."""

    def _load_model(self):
        # Note: SentenceTransformer has to be loaded as AutoModel as torch.compile does not work for SentenceTransformer
        self.model_class = self._get_model_class(self.model_path)
        self.model = self.model_class.from_pretrained(self.model_path)

    def _prep_input(self):
        """Prepare input tensors for Hugging Face models."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_id = self.tokenizer(
            [self.prompt], padding=True, truncation=True, return_tensors="pt"
        )

    def _generate_output(self, safe_warmup):
        """Generate output using the model's generate method."""
        print(f"Generating output for {self.model_class} on {self.device}...")

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
    
    def _forward_output(self):
        if self.model_class in ["causal_lm"]:
            return self.model(self.input_id["input_ids"])
        else:
            return self.model(self.input_id)
