class ModelHandlerFMS(ModelHandlerBase):
    """Model handler for FMS models."""
    def _load_model(self):
         # This get_model call assumes locally downloaded weights
            self.model = get_model(
                "hf_pretrained",
                variant=self.model_path,
                device_type="cpu",
                data_type=torch.float16,
                fused_weights=False,
            )

    def prep_input(self):
        """Prepares input for the model."""
        self.tokenizer = tokenizers.get_tokenizer(self.model_path)
        tokens = self.tokenizer.tokenize(self.prompt)
        ids_l = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.tokenizer.bos_token_id != self.tokenizer.eos_token_id:
            ids_l = [self.tokenizer.bos_token_id] + ids_l

        prompt1 = torch.tensor(ids_l, dtype=torch.long, device=self.device)
        self.input_id, self.extra_generation_kwargs = pad_input_ids(
            [prompt1], min_pad_length=self.min_pad_length
        )