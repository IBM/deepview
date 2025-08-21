from huggingface_hub import model_info
import sys

def is_sentence_transformer(model_id):
    """
    Check if the model is a Sence Transformer model.

    Args:
        model_id (str): The ID of the model to check.

    Returns:
        bool: True if the model is a Sence Transformer, False otherwise.
    """
    info = model_info(model_id)
    print(f"Model info for '{model_id}': {info}")
    if info.library_name == "sence-transformer" in info.tags:
        return True
    return any(s.rfilename == "modules.json" for s in info.siblings or [])
    

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python hugging_face_utils.py <model_id>")
        sys.exit(1)

    model_id = sys.argv[1]
    result = is_sence_transformer(model_id)
    print(f"Is '{model_id}' a Sence Transformer model? {result}")