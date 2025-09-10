import pickle
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import warnings
import json

try:
    from safetensors.torch import save_file as safetensors_save, load_file as safetensors_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    warnings.warn("safetensors not found. Install with: pip install safetensors")


def save_data(data: Any, filepath: Union[str, Path], use_safetensors: bool = True) -> None:
    """
    Save a data structure that may contain PyTorch tensors to disk.
    
    Args:
        data: Any Python data structure (dict, list, tuple, etc.) that may contain PyTorch tensors
        filepath: Path where to save the data (without extension)
        use_safetensors: Whether to use safetensors for tensor storage (recommended)
    """
    filepath = Path(filepath)
    
    # Extract tensors and their paths, plus create structure without tensors
    tensors_dict, structure = _extract_tensors(data)
    
    if tensors_dict:
        # Save tensors using safetensors or torch.save
        if use_safetensors and HAS_SAFETENSORS:
            safetensors_save(tensors_dict, str(filepath) + '_tensors.safetensors')
        else:
            torch.save(tensors_dict, str(filepath) + '_tensors.pt')
    
    # Save the structure (without tensors) using pickle
    with open(str(filepath) + '_structure.pkl', 'wb') as f:
        pickle.dump(structure, f)
    
    # Save metadata
    metadata = {
        'has_tensors': bool(tensors_dict),
        'tensor_format': 'safetensors' if (use_safetensors and HAS_SAFETENSORS) else 'torch',
        'tensor_paths': list(tensors_dict.keys()) if tensors_dict else []
    }
    
    with open(str(filepath) + '_metadata.json', 'w') as f:
        json.dump(metadata, f)


def load_data(filepath: Union[str, Path]) -> Any:
    """
    Load a data structure that may contain PyTorch tensors from disk.
    
    Args:
        filepath: Path to the saved data file (without extension)
        
    Returns:
        The original data structure with PyTorch tensors restored
    """
    filepath = Path(filepath)
    
    # Load metadata
    with open(str(filepath) + '_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load structure
    with open(str(filepath) + '_structure.pkl', 'rb') as f:
        structure = pickle.load(f)
    
    # Load tensors if they exist
    tensors_dict = {}
    if metadata['has_tensors']:
        if metadata['tensor_format'] == 'safetensors':
            tensors_dict = safetensors_load(str(filepath) + '_tensors.safetensors')
        else:
            tensors_dict = torch.load(str(filepath) + '_tensors.pt', map_location='cpu')
    
    # Reconstruct the original data structure
    return _reconstruct_data(structure, tensors_dict)


def _extract_tensors(obj: Any, path: str = "") -> Tuple[Dict[str, torch.Tensor], Any]:
    """
    Recursively extract PyTorch tensors from a data structure.
    
    Returns:
        tensors_dict: Dictionary mapping paths to tensors
        structure: Original structure with tensors replaced by placeholders
    """
    tensors_dict = {}
    
    if torch.is_tensor(obj):
        tensor_path = f"tensor_{path}" if path else "tensor_root"
        tensors_dict[tensor_path] = obj
        
        # Store tensor metadata for reconstruction
        placeholder = {
            '_tensor_placeholder': tensor_path,
            '_tensor_device': str(obj.device),
            '_tensor_dtype': str(obj.dtype),
            '_tensor_requires_grad': obj.requires_grad,
            '_tensor_shape': list(obj.shape)
        }
        return tensors_dict, placeholder
    
    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            sub_tensors, sub_structure = _extract_tensors(value, f"{path}.{key}" if path else str(key))
            tensors_dict.update(sub_tensors)
            new_dict[key] = sub_structure
        return tensors_dict, new_dict
    
    elif isinstance(obj, list):
        new_list = []
        for i, item in enumerate(obj):
            sub_tensors, sub_structure = _extract_tensors(item, f"{path}[{i}]" if path else f"[{i}]")
            tensors_dict.update(sub_tensors)
            new_list.append(sub_structure)
        return tensors_dict, new_list
    
    elif isinstance(obj, tuple):
        new_items = []
        for i, item in enumerate(obj):
            sub_tensors, sub_structure = _extract_tensors(item, f"{path}({i})" if path else f"({i})")
            tensors_dict.update(sub_tensors)
            new_items.append(sub_structure)
        return tensors_dict, tuple(new_items)
    
    elif isinstance(obj, set):
        # Sets are tricky with tensors since tensors aren't hashable
        # Convert to list, extract, then convert back
        new_items = []
        for i, item in enumerate(obj):
            sub_tensors, sub_structure = _extract_tensors(item, f"{path}{{{i}}}" if path else f"{{{i}}}")
            tensors_dict.update(sub_tensors)
            new_items.append(sub_structure)
        return tensors_dict, {'_set_placeholder': new_items}
    
    else:
        # For other types (int, float, str, etc.), return as-is
        return tensors_dict, obj


def _reconstruct_data(structure: Any, tensors_dict: Dict[str, torch.Tensor]) -> Any:
    """
    Recursively reconstruct the original data structure by replacing placeholders with tensors.
    """
    if isinstance(structure, dict):
        if '_tensor_placeholder' in structure:
            # This is a tensor placeholder - restore with original properties
            tensor_path = structure['_tensor_placeholder']
            tensor = tensors_dict[tensor_path]
            
            # Restore original device
            original_device = structure['_tensor_device']
            try:
                if str(tensor.device) != original_device:
                    tensor = tensor.to(original_device)
            except RuntimeError as e:
                warnings.warn(f"Could not restore tensor to device {original_device}: {e}. "
                            f"Keeping on {tensor.device}")
            
            # Restore requires_grad
            original_requires_grad = structure['_tensor_requires_grad']
            if tensor.requires_grad != original_requires_grad:
                tensor = tensor.requires_grad_(original_requires_grad)
            
            # Verify shape (should match but good to check)
            original_shape = structure['_tensor_shape']
            if list(tensor.shape) != original_shape:
                warnings.warn(f"Tensor shape mismatch: expected {original_shape}, got {list(tensor.shape)}")
            
            return tensor
        elif '_set_placeholder' in structure:
            # This is a set placeholder
            reconstructed_items = [_reconstruct_data(item, tensors_dict) for item in structure['_set_placeholder']]
            return set(reconstructed_items)
        else:
            # Regular dictionary
            return {key: _reconstruct_data(value, tensors_dict) for key, value in structure.items()}
    
    elif isinstance(structure, list):
        return [_reconstruct_data(item, tensors_dict) for item in structure]
    
    elif isinstance(structure, tuple):
        return tuple(_reconstruct_data(item, tensors_dict) for item in structure)
    
    else:
        # For other types, return as-is
        return structure


def save_data_simple(data: Any, filepath: Union[str, Path]) -> None:
    """
    Simple version that saves everything in one file using torch.save.
    Good for quick prototyping but less secure than safetensors.
    
    Args:
        data: Any Python data structure that may contain PyTorch tensors
        filepath: Path where to save the data
    """
    filepath = Path(filepath)
    torch.save(data, filepath)


def load_data_simple(filepath: Union[str, Path]) -> Any:
    """
    Simple version that loads everything from one file using torch.load.
    
    Args:
        filepath: Path to the saved data file
        
    Returns:
        The original data structure
    """
    filepath = Path(filepath)
    return torch.load(filepath, map_location='cpu')


# Example usage and testing
if __name__ == "__main__":
    # Create test data with mixed types and tensors on different devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = {
        'tensor1': torch.randn(3, 4).to(device),
        'tensor2': torch.tensor([1, 2, 3], dtype=torch.int64),
        'nested': {
            'tensor3': torch.randn(2, 2, requires_grad=True).to(device),
            'list_with_tensor': [torch.randn(1, 3, dtype=torch.float16), 'string', 42],
            'normal_data': [1, 2, 3, 'hello']
        },
        'tuple_data': (torch.randn(2,), 'test', {'inner_tensor': torch.randn(1,)}),
        'normal_list': [1, 2, 3],
        'normal_dict': {'a': 1, 'b': 2}
    }
    
    print("Testing safetensors version:")
    # Save the data using safetensors
    save_data(test_data, 'test_data_safe', use_safetensors=True)
    print("Data saved successfully with safetensors!")
    
    # Load the data back
    loaded_data = load_data('test_data_safe')
    print("Data loaded successfully!")
    
    # Verify the structure is preserved
    print(f"Original keys: {list(test_data.keys())}")
    print(f"Loaded keys: {list(loaded_data.keys())}")
    
    # Check tensor shapes and properties
    print(f"Original tensor1 shape: {test_data['tensor1'].shape}")
    print(f"Loaded tensor1 shape: {loaded_data['tensor1'].shape}")
    print(f"Original tensor1 device: {test_data['tensor1'].device}")
    print(f"Loaded tensor1 device: {loaded_data['tensor1'].device}")
    print(f"Original tensor1 dtype: {test_data['tensor1'].dtype}")
    print(f"Loaded tensor1 dtype: {loaded_data['tensor1'].dtype}")
    print(f"Tensor3 requires_grad: {loaded_data['nested']['tensor3'].requires_grad}")
    print(f"List tensor dtype: {loaded_data['nested']['list_with_tensor'][0].dtype}")
    print(f"Devices match: {test_data['tensor1'].device == loaded_data['tensor1'].device}")
    print(f"Dtypes match: {test_data['tensor1'].dtype == loaded_data['tensor1'].dtype}")
    print(f"Requires_grad match: {test_data['nested']['tensor3'].requires_grad == loaded_data['nested']['tensor3'].requires_grad}")
    
    print("\nTesting simple version:")
    # Test simple version
    save_data_simple(test_data, 'test_data_simple.pt')
    loaded_simple = load_data_simple('test_data_simple.pt')
    print("Simple version works too!")
    
    # Verify tensors are identical
    print(f"Tensors equal: {torch.equal(test_data['tensor1'], loaded_data['tensor1'])}")







    