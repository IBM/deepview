from deepview.utils.ModelHandler.model_handler import ModelHandlerBase


class ModelHandlerFMS(ModelHandlerBase):
    """Handles FMS models with specific input preparation and compilation methods."""

    def _load_model(self):
        self.model = get_model(
            "hf_pretrained",
            variant=self.model_path,
            device_type="cpu",
            data_type=torch.float16,
            fused_weights=False,
        )
        
    def _prep_input(self):
        """Prepare input tensors for FMS models."""
        self.tokenizer = tokenizers.get_tokenizer(self.model_path)
        tokens = self.tokenizer.tokenize(self.prompt)
        ids_l = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.tokenizer.bos_token_id != self.tokenizer.eos_token_id:
            ids_l = [self.tokenizer.bos_token_id] + ids_l

        prompt1 = torch.tensor(ids_l, dtype=torch.long, device=self.device)
        self.input_id, self.extra_generation_kwargs = pad_input_ids(
            [prompt1], min_pad_length=self.min_pad_length
        )
        
    def _generate_output(self, is_warmup):
        """Generate output using the model's generate method."""
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