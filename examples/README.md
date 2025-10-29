# Using Deepview as a Library
This document walks through how to call the core modules of DeepView from within your own inference script. This allows us to debug and analyze a larger set of models that deepview does not have support for yet in the CLI tool. 

## Unsupported Ops Mode
To use unsupported ops mode within your inference script, you will need the following steps:

**1. Add Imports**
```python
import torch_sendnn
from deepview.core.unsupported_ops import process_unsupported_ops
```

**2. Prep Model Input**
This step will look different for every model, ensure that you have the proper setup for input in your script

**3. Load and Compile Model** 
There are 2 ways to load a model:
1. Using Hugging Face Transformers library (most-common)
2. Using Foundation Model Stack (model architecture must be supported)

Loading the model is different for each one, ensure that you use the proper loader in your script

After you load the model, you must also add the following compile line (this ensures you are running on Spyre):

```python
model.compile(backend="sendnn")
```

**4. Call Generate on the Model**
This step is dependent on the input prep required for your model and how the model is loaded.
1. If you load with Hugging Face Transformers, you can call `model.generate` and pass any input / kwargs needed
2. If you load with Foundation Model Stack, FMS provides a generate utility you can import directly into your script:
```
from fms.utils.generation import generate, pad_input_ids
```

Ensure to wrap your generate call in the following block:
```python
with torch_sendnn.warmup_mode(skip_compilation=True):
    model.generate(...)
```

**5. Process Unsupported Ops with DeepView**
The process unsupported ops method takes 2 arguments:
1. show_details_flag (bool): Whether to print detailed stack traces for each unsupported op.
2. generate_repro_code_flag (bool): Whether to generate minimal repro scripts for each op.

Add this line to your script with your desired values for the two flags:

```python
process_unsupported_ops(True, False) 
```

#### Example Scripts 
Example scripts are provided in the `unsupported_ops/` folder located in this directory

Run the scripts:
```
python3 unsupported_ops/unsupp_ops_bamba.py
```
```
python3 unsupported_ops/unsupp_ops_gvision_fms.py
```

The following script uses the recipe above but will **NOT** return successfully. This is the current status of this model and needs further investigation
```
python3 unsupported_ops/unsupp_ops_smolVLM-256M-Instruct.py
```

## Layer Debugging Mode
Coming Soon

## Layer IO Divergence Mode 
Coming Soon 