# Third Party
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
import torch
import torch_sendnn

# Local
from deepview.core.unsupported_ops import process_unsupported_ops

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image(
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
)

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"},
        ],
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Compile model for sendnn backend
model.compile(backend="sendnn")

# Generate outputs
with torch_sendnn.warmup_mode(skip_compilation=True):
    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

print(generated_texts[0])

# Process Unsupported Ops
process_unsupported_ops(True, True)
