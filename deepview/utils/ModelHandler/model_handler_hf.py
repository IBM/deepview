from deepview.utils.ModelHandler.model_handler import ModelHandlerBase


class ModelHandlerHF(ModelHandlerBase):
    """Handles Hugging Face models with specific input preparation and compilation methods."""
    
    def _load_model(self):
        # Note: SentenceTransformer has to be loaded as AutoModel as torch.compile does not work for SentenceTransformer
        self.model_class = self._get_model_class(self.model_path)
        self.model = self.model_class.from_pretrained(self.model_path)

    def _prep_input(self):
        """Prepare input tensors for Hugging Face models."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_id = self.tokenizer(
            [self.prompt], padding=True, truncation=True, return_tensors="pt"
        )
    
    def _generate_output(self, is_warmup):
        """Generate output using the model's generate method."""
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
        return result