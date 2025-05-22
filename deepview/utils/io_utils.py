import os 
import torch 
import inspect
import copy
import pickle
import torch_sendnn 
from torch_sendnn.backends import sendnn_backend, lazy_handles
from pycony import *


def save_ios(layer_name, inputs, outputs, base_dir="saved_artifacts", save_dir=None):
    """
    Save the inputs and outputs of a module, using torch.save for tensors.
    
    Args:
        layer_name: Name of the layer
        inputs: Input tensors to the module
        outputs: Output tensors from the module
        base_dir: Base directory to save the data
    
    Returns:
        Dictionary with paths to saved files
    """
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a clean filename from layer_name
    clean_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '_').replace('__', '_')

    # save_dir = os.path.join(base_dir, clean_name)

    # Find the next available iteration number
    if save_dir is None:
        i = 0
        while True:
            save_dir = os.path.join(base_dir, f"{clean_name}_{i}")
            if not os.path.exists(save_dir):
                break
            i += 1
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Helper function to process and save tensors
    def process_and_save(data, prefix):
        structure = {}  # To store structural information
        
        def _process(obj, path=""):
            if isinstance(obj, torch.Tensor):
                # For tensors, save to file using torch.save
                tensor_path = os.path.join(save_dir, f"{prefix}_{path}.pt")
                tensor = obj.detach().cpu()
                torch.save(tensor, tensor_path)
                # Record tensor information in structure
                structure[path] = {
                    "type": "tensor",
                    "path": tensor_path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype)
                }
                return f"TENSOR_REF:{path}"
            
            elif isinstance(obj, list):
                # Process each item in the list
                for i, item in enumerate(obj):
                    new_path = f"{path}_{i}" if path else f"{i}"
                    _process(item, new_path)
                # Record list structure
                structure[path] = {
                    "type": "list",
                    "length": len(obj),
                    "items": [f"{path}_{i}" if path else f"{i}" for i in range(len(obj))]
                }
                
            elif isinstance(obj, tuple):
                # Process each item in the tuple
                for i, item in enumerate(obj):
                    new_path = f"{path}_{i}" if path else f"{i}"
                    _process(item, new_path)
                # Record tuple structure
                structure[path] = {
                    "type": "tuple",
                    "length": len(obj),
                    "items": [f"{path}_{i}" if path else f"{i}" for i in range(len(obj))]
                }
                
            elif isinstance(obj, dict):
                # Process each item in the dict
                for key, value in obj.items():
                    # Use string keys in the path
                    safe_key = str(key).replace('.', '_').replace('/', '_')
                    new_path = f"{path}_{safe_key}" if path else safe_key
                    _process(value, new_path)
                # Record dict structure
                structure[path] = {
                    "type": "dict",
                    "keys": list(obj.keys()),
                    "items": {key: f"{path}_{key}" if path else key for key in obj.keys()}
                }
                
            else:
                # For non-tensor objects, just store their value in the structure
                structure[path] = {
                    "type": "value",
                    "value": obj
                }
        
        # Process the data
        _process(data)
        
        # Save the structure
        structure_path = os.path.join(save_dir, f"{prefix}_structure.pkl")
        with open(structure_path, 'wb') as f:
            pickle.dump(structure, f)
            
        return structure_path
    
    # Save inputs and outputs
    input_structure_path = process_and_save(inputs, "input")
    output_structure_path = process_and_save(outputs, "output")
    
    # Save metadata
    metadata = {
        "layer_name": layer_name,
        "input_structure_path": input_structure_path,
        "output_structure_path": output_structure_path,
        "timestamp": torch.cuda.Event(enable_timing=True).record() if torch.cuda.is_available() else None
    }
    
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return {
        "save_dir": save_dir,
        "metadata_path": metadata_path,
        "input_structure_path": input_structure_path,
        "output_structure_path": output_structure_path
    }

def load_ios(layer_name, index=0, base_dir="saved_artifacts"):
    """
    Load the inputs and outputs for a module.
    
    Args:
        layer_name: Name of the layer
        base_dir: Base directory where the data was saved
    
    Returns:
        Dictionary containing inputs and outputs
    """
    # Create a clean filename from layer_name
    clean_name = layer_name.replace('.', '_').replace('[', '_').replace(']', '_').replace('__', '_')
    save_dir = os.path.join(base_dir, clean_name)
    # if not os.path.exists(save_dir):
    save_dir = save_dir+'_'+str(index)
    print(f"Loading saved inputs and outputs from {save_dir}")
    metadata_path = os.path.join(save_dir, "metadata.pkl")

    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: Could not find saved I/O metadata for {layer_name}")
        return None
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Helper function to reconstruct objects from structure
    def reconstruct(structure_path, prefix):
        # Load structure
        with open(structure_path, 'rb') as f:
            structure = pickle.load(f)
            
        # Recursive function to build objects from structure
        def _build(path=""):
            if path not in structure:
                print(f"Warning: Path '{path}' not found in structure")
                return None
                
            info = structure[path]
            obj_type = info["type"]
            
            if obj_type == "tensor":
                # Load tensor from file
                tensor_path = info["path"]
                if os.path.exists(tensor_path):
                    return torch.load(tensor_path, weights_only=True)
                else:
                    print(f"Warning: Tensor file not found: {tensor_path}")
                    # Create a dummy tensor with the correct shape and dtype
                    dtype_map = {
                        "torch.float32": torch.float32,
                        "torch.float": torch.float,
                        "torch.float64": torch.float64,
                        "torch.double": torch.double,
                        "torch.float16": torch.float16,
                        "torch.half": torch.half,
                        "torch.int32": torch.int32,
                        "torch.int": torch.int,
                        "torch.int64": torch.int64,
                        "torch.long": torch.long,
                        "torch.bool": torch.bool
                    }
                    dtype = dtype_map.get(info["dtype"], torch.float32)
                    return torch.zeros(info["shape"], dtype=dtype)
            
            elif obj_type == "list":
                # Rebuild list
                return [_build(item) for item in info["items"]]
            
            elif obj_type == "tuple":
                # Rebuild tuple
                return tuple(_build(item) for item in info["items"])
            
            elif obj_type == "dict":
                # Rebuild dict
                keys = info["keys"]
                items = info["items"]
                return {k: _build(items[k]) for k in keys}
            
            elif obj_type == "value":
                # Return stored value
                return info["value"]
            
            else:
                print(f"Warning: Unknown object type '{obj_type}'")
                return None
        
        # Start reconstruction from root
        return _build()
    
    # Reconstruct inputs and outputs
    inputs = reconstruct(metadata["input_structure_path"], "input")
    outputs = reconstruct(metadata["output_structure_path"], "output")
    
    return {
        "layer_name": metadata["layer_name"],
        "inputs": inputs,
        "outputs": outputs,
        "timestamp": metadata.get("timestamp")
    }

def load_ios_from_dir(save_dir, base_dir="saved_artifacts"):
    """
    Load the inputs and outputs for a module from save_dir
    
    Args:
        save_dir : Artifacts directory 
        base_dir: Base directory where the data was saved
    
    Returns:
        Dictionary containing inputs and outputs
    """
    metadata_path = os.path.join(save_dir, "metadata.pkl")
    print(f"Loading saved inputs and outputs from {save_dir}")
    # Check if metadata file exists
    if not os.path.exists(metadata_path):
        print(f"Error: Could not find saved I/O metadata for {layer_name}")
        return None
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Helper function to reconstruct objects from structure
    def reconstruct(structure_path, prefix):
        # Load structure
        with open(structure_path, 'rb') as f:
            structure = pickle.load(f)
            
        # Recursive function to build objects from structure
        def _build(path=""):
            if path not in structure:
                print(f"Warning: Path '{path}' not found in structure")
                return None
                
            info = structure[path]
            obj_type = info["type"]
            
            if obj_type == "tensor":
                # Load tensor from file
                tensor_path = info["path"]
                if os.path.exists(tensor_path):
                    return torch.load(tensor_path, weights_only=True)
                else:
                    print(f"Warning: Tensor file not found: {tensor_path}")
                    # Create a dummy tensor with the correct shape and dtype
                    dtype_map = {
                        "torch.float32": torch.float32,
                        "torch.float": torch.float,
                        "torch.float64": torch.float64,
                        "torch.double": torch.double,
                        "torch.float16": torch.float16,
                        "torch.half": torch.half,
                        "torch.int32": torch.int32,
                        "torch.int": torch.int,
                        "torch.int64": torch.int64,
                        "torch.long": torch.long,
                        "torch.bool": torch.bool
                    }
                    dtype = dtype_map.get(info["dtype"], torch.float32)
                    return torch.zeros(info["shape"], dtype=dtype)
            
            elif obj_type == "list":
                # Rebuild list
                return [_build(item) for item in info["items"]]
            
            elif obj_type == "tuple":
                # Rebuild tuple
                return tuple(_build(item) for item in info["items"])
            
            elif obj_type == "dict":
                # Rebuild dict
                keys = info["keys"]
                items = info["items"]
                return {k: _build(items[k]) for k in keys}
            
            elif obj_type == "value":
                # Return stored value
                return info["value"]
            
            else:
                print(f"Warning: Unknown object type '{obj_type}'")
                return None
        
        # Start reconstruction from root
        return _build()
    
    # Reconstruct inputs and outputs
    inputs = reconstruct(metadata["input_structure_path"], "input")
    outputs = reconstruct(metadata["output_structure_path"], "output")
    
    return {
        "layer_name": metadata["layer_name"],
        "inputs": inputs,
        "outputs": outputs,
        "timestamp": metadata.get("timestamp")
    }

def save_layer_stack(layer_stack, base_dir="saved_artifacts", filename="model_layers_stack.pkl"):
  """
  Pickles the given list `layer_stack` to a file in the specified base directory.

  Args:
      layer_stack: The list to be pickled.
      base_dir: The base directory where the file will be saved. Defaults to "saved_artifacts".
      filename: The name of the pickle file. Defaults to "model_layers_stack.pkl".
  """
  os.makedirs(base_dir, exist_ok=True)
  filepath = os.path.join(base_dir, filename)
  try:
    with open(filepath, 'wb') as f:
      pickle.dump(layer_stack, f)
    print(f"Layer stack pickled successfully to: {filepath}")
  except Exception as e:
    print(f"Error pickling layer stack to {filepath}: {e}")

def load_layer_stack(base_dir="saved_artifacts", filename="model_layers_stack.pkl"):
  """
  Loads a pickled layer stack from the specified file.

  Args:
      base_dir: The base directory where the pickle file is located. Defaults to "saved_artifacts".
      filename: The name of the pickle file. Defaults to "model_layers_stack.pkl".

  Returns:
      The loaded layer stack (a Python list), or None if the file is not found or an error occurs during loading.
  """
  filepath = os.path.join(base_dir, filename)
  try:
    with open(filepath, 'rb') as f:
      loaded_layer_stack = pickle.load(f)
    print(f"Layer stack loaded successfully from: {filepath}")
    return loaded_layer_stack
  except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    return None
  except Exception as e:
    print(f"Error loading layer stack from {filepath}: {e}")
    return None

def compare_tensor_tuples(tuple1, tuple2):
    """
    Compares two tuples of tensors for equality.
    Args:
        tuple1: The first tuple of tensors.
        tuple2: The second tuple of tensors.
    Returns:
        True if the tuples are equal, False otherwise.
    """
    # Check if tuples have the same length
    if len(tuple1) != len(tuple2):
        return False
    # Iterate through the elements and compare tensors
    for i, (tensor1, tensor2) in enumerate(zip(tuple1, tuple2)):
        if not torch.equal(tensor1, tensor2):
            return False
    # If all tensors are equal, return True
    return True

def compare_tensor_tuples_stats(tuple1, tuple2):
    """
    Calculates the mean, median, Q1, and Q3 of the absolute difference
    between two tuples of tensors.

    Args:
        tuple1: The first tuple of tensors.
        tuple2: The second tuple of tensors.

    Returns:
        A dictionary containing the mean, median, Q1, and Q3 of the
        absolute differences. Returns None if the input tuples have
        different lengths or contain non-tensor elements.
    """
    if len(tuple1) != len(tuple2):
        return None
    absolute_differences = []
    for tensor1, tensor2 in zip(tuple1, tuple2):     
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            return None
        absolute_diff = torch.abs(tensor1 - tensor2).flatten().tolist()
        absolute_differences.extend(absolute_diff)
        print("DEBUG: compare_tensor_tuples_stats")

    if len(absolute_differences) == 0:
        return {"mean": float('nan'), "median": float('nan'), "q1": float('nan'), "q3": float('nan')}

    abs_diff_tensor = torch.tensor(absolute_differences)
    abs_diff_tensor = torch.nan_to_num(abs_diff_tensor, nan=0.0) 
    mean_diff = torch.mean(abs_diff_tensor).item()
    median_diff = torch.median(abs_diff_tensor).item()

    q1_diff = torch.quantile(abs_diff_tensor, 0.25).item()
    q3_diff = torch.quantile(abs_diff_tensor, 0.75).item()

    return {
        "mean": mean_diff,
        "median": median_diff,
        "q1": q1_diff,
        "q3": q3_diff
    }

def compile_and_run_layer(model, save_dir):
    """
    Compiles and runs a specified layer with saved inputs, then compares its output
    to expected outputs.

    Args:
        model: The model containing the layer to be tested. (Though 'model' isn't
               directly used in the provided snippet, it's included as a parameter
               if it's part of a larger context where the layer might be extracted
               from it.)
        save_dir (str): The directory where the layer's inputs and outputs are saved.

    Returns:
        bool: True if the actual output matches the expected output, False otherwise.
    """
    # Load saved input/output data for the layer
    io_data = load_ios_from_dir(save_dir)
    layer_name_str = io_data['layer_name']
    print(f"===== Testing: {layer_name_str} =====")

    target_layer = eval(layer_name_str)
    
    # Get the expected arguments for the layer's forward method
    forward_signature = inspect.signature(target_layer.forward)
    expected_args = list(forward_signature.parameters.keys())
    
    # Retrieve input values from saved data
    input_values = list(io_data['inputs'])

    print(f"Expected Arguments: {expected_args}")
    print("Input Values:")
    for i, val in enumerate(input_values):
        if isinstance(val, torch.Tensor):
            print(f"  [{i}]: Tensor of shape {val.shape}")
        else:
            print(f"  [{i}]: {val}")

    if len(input_values) < len(expected_args):
        zipped_inputs = list(zip_longest(expected_args, input_values, fillvalue=None))
    else:
        zipped_inputs = list(zip(expected_args, input_values))
    
    kwargs = dict(zipped_inputs)
    print(f"Input Keyword Arguments: {kwargs}")

    # Compile the layer for AIU
    compiled_layer = torch.compile(target_layer, backend='sendnn')
    # Run the compiled layer in warm-up mode 
    with torch_sendnn.warmup_mode():
        actual_output = compiled_layer(**kwargs)
    
    expected_output = io_data['outputs']

    print(f"Actual Output: {actual_output}")
    print(f"Expected Output: {expected_output}")
    
    # Print comparison statistics for debugging/analysis
    print(compare_tensor_tuples_stats(actual_output, expected_output))
    
    # Return a boolean indicating whether the outputs match
    return compare_tensor_tuples(actual_output, expected_output)

