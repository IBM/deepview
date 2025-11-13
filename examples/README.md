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
**1. ibm-ai-platform/Bamba-9B-v2**
```
python3 unsupported_ops/unsupp_ops_bamba.py
```
Expected Output:
```
DEEPVIEW========================================================================
DEEPVIEW Unsupported operations list:
DEEPVIEW copy
DEEPVIEW le
DEEPVIEW========================================================================
```

**2. ibm-granite/granite-vision-3.3-2b**
```
python3 unsupported_ops/unsupp_ops_gvision_fms.py
```

Expected Output: 
```
<|system|>
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
<|user|>

What animal is shown in this image?
<|assistant|>

DEEPVIEW========================================================================
DEEPVIEW No unsupported operations detected.
DEEPVIEW========================================================================
```

**3. HuggingFaceTB/SmolVLM-256M-Instruct**
The following script uses the recipe above but will **NOT** return successfully. This is the current status of this model and needs further investigation
```
python3 unsupported_ops/unsupp_ops_smolVLM-256M-Instruct.py
```

Expected Output:
```
/home/senuser/.local/lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py:2242: FutureWarning: The class `AutoModelForVision2Seq` is deprecated and will be removed in v5.0. Please use `AutoModelForImageTextToText` instead.
  warnings.warn(
`torch_dtype` is deprecated! Use `dtype` instead!
Traceback (most recent call last):
  File "/home/senuser/deepview/examples/unsupported_ops/unsupp_ops_smolVLM-256M-Instruct.py", line 47, in <module>
    generated_ids = model.generate(**inputs, max_new_tokens=500)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/generation/utils.py", line 2539, in generate
    result = self._sample(
             ^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/generation/utils.py", line 2867, in _sample
    outputs = self(**model_inputs, return_dict=True)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1749, in _wrapped_call_impl
    return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 655, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/utils/generic.py", line 940, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 973, in forward
    outputs = self.model(
              ^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/utils/generic.py", line 940, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 795, in forward
    image_hidden_states = self.get_image_features(pixel_values, pixel_attention_mask)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 702, in get_image_features
    pixel_values = pixel_values[real_images_inds].contiguous()
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 714, in torch_dynamo_resume_in_get_image_features_at_702
    pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 722, in torch_dynamo_resume_in_get_image_features_at_714
    image_hidden_states = self.vision_model(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 561, in forward
    hidden_states = self.embeddings(pixel_values=pixel_values, patch_attention_mask=patch_attention_mask)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib64/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/senuser/.local/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py", line 164, in forward
    position_ids[batch_idx][p_attn_mask.view(-1)] = pos_ids
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape mismatch: value tensor of shape [594] cannot be broadcast to indexing result of shape [466]
```

## Layer Debugging Mode
Coming Soon

## Layer IO Divergence Mode 
Coming Soon 