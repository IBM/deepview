class ModelHandlerHF(ModelHandler):
    """Handles loading, compiling, input preparation, inference, and debugging for Hugging Face models.

    Inherits from ModelHandler and specializes for Hugging Face model types.
    """

    def __init__(self, model_path, device, prompt):
        super().__init__(
            model_type="hf",
            model_path=model_path,
            device=device,
            prompt=prompt,
        )
    def load_model(self):
        """Loads the Hugging Face model from the specified path."""
        print("Loading Hugging Face model...")
        # TODO: we can do specific handling per model class but for now everything apart from CausalLM is treated as AutoModel
        # Note: SentenceTransformer has to be loaded as AutoModel as torch.compile does not work for SentenceTransformer
        self.model_class = self._infer_model_class(self.model_path)
        print("---------------------------------------------------------")
        print(f"Model class is {self.model_class}")
        print("---------------------------------------------------------")
        self.model = self.model_class.from_pretrained(self.model_path)
              
    def compile_model(self):
        """Compiles the Hugging Face model for inference."""
        print("Compiling Hugging Face model...")
    
    def prepare_input(self, input_data):
        """Prepares input data for the Hugging Face model."""
        print("Preparing input for Hugging Face model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_id = self.tokenizer(
            [self.prompt], padding=True, truncation=True, return_tensors="pt"
        )
    
    def _generate_output(self):
        """Generates output from the Hugging Face model."""
        print("Generating output from Hugging Face model...")
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
