from .model_handler import ModelHandlerBase
import torch
from fms.models import get_model
from fms.utils import tokenizers
from fms.utils.generation import generate, pad_input_ids

class ModelHandlerFMS(ModelHandlerBase):
    """Handles FMS models with specific input preparation and compilation methods."""

    def _load_model(self):
        print(f"Loading FMS model from {self.model_path}...")
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