
<p align="center">
  <picture>
    <img alt="Spyer AIU DeepView" src="./logo/deepview-logo.svg" width="250" height="259" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-Apache%202.0-blue?color=blue"></a>
    <a href="./LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/IBM-Internal-red?color=red"></a>
    <a href="./LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/in_progress-green"></a>
</p>

# DeepView 
DeepView is a diagnostic tool to accelerate the model enablement journey on AIU. It identifies issues during model lowering and pinpoints root causes with minimal granularity.

In the current version, DeepView identifies unsupported ops with detailed metadata (such as input shapes and dtypes). If enabled, it also provides precise stack traces to locate the unsupported op in the model source code. This empowers quick root-cause analysis during model enablement journey.

The upcoming version of DeepView will expand its capabilities to automatically generate minimal reproducible scripts for unsupported ops.
It will also go beyond unsupported ops to capture more complex issues, such as internal compiler assertion failures, and precisely identify the offending layer within the transformer blocks.

# Installation
### local install
```shell
pip install -e .
```
or 
```shell
python setup.py install
```

# Usage
> [!NOTE]
> Please note that the instructions for torch_sendnn given below are temporary. We are working with torch_sendnn to get these changes incorporated into it.

First, copy `torch_sendnn` from its installation directory to `/tmp`:

If you are using the `e2e-stable` image, the installation directory of `torch_sendnn` is typically `/usr/local/lib/python3.12/site-packages/torch_sendnn`. Otherwise, you may use `python3 -m pip show torch_sendnn` to find out the installation directory.
Replace the `/tmp/torch_sendnn/backends.py` and `/tmp/torch_sendnn/torch_sendnn.py` files with [deepview/core/tmp/backends.py](/core/tmp/backends.py) and [deepview/core/tmp/torch_sendnn.py](/core/tmp/torch_sendnn.py) files, respectively, given in this repository.

Next, set the PYTHONPATH.
```
export PYTHONPATH=/tmp:$PYTHONPATH
```

Now, run deepview as follows.

```
Usage: python3 deepview [-h] --model_type {fms,hf} --model MODEL --mode {unsupported_op} [{unsupported_op} ...] [--show_details] --output_file OUTPUT_FILE

Script to run DeepView tool on any model.

options:
  -h, --help            show this help message and exit
  --model_type {fms,hf}
                        The type of model you want to debug - fms or hf.
  --model MODEL         Model name in HF format or model path
  --mode {unsupported_op, layer_debugging} [{unsupported_op, layer_debugging} ...]
                        Modes: [unsupported_op, layer_debugging] (Choose one or more). Default is the unsupported_op mode.
  --show_details        Print stack trace and other details, valid only with unsupported_op.
  --output_file OUTPUT_FILE
                        Name of the file in which the debug tool output will be stored.
```

## Examples

### unsupported_op mode

```
python3 deepview.py --model_type fms --model /mnt/aiu-models-en-shared/models/ibm-ai-platform/Bamba-9B --mode unsupported_op --show_details --output_file debugger.txt
```

#### Sample Output

```
DEBUG TOOL Caught error for constant_pad_nd_81: Operation not supported.
DEBUG TOOL Data type: sen_datatype_enum.float32, Shape: [ 1 256 128 128 ]
DEBUG TOOL==================================== Stack Trace ====================================
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/models/bamba.py", line 448, in forward
DEBUG TOOL     output, cache = self.base_model(
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/models/bamba.py", line 365, in forward
DEBUG TOOL     output = layer(
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/models/bamba.py", line 145, in forward
DEBUG TOOL     x = self.ssm(
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/modules/ssm.py", line 396, in forward
DEBUG TOOL     hidden_states, A, B, C = [
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/modules/ssm.py", line 397, in <listcomp>
DEBUG TOOL     reshape_into_chunks(t, pad_size, self.chunk_size)
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/modules/ssm.py", line 32, in reshape_into_chunks
DEBUG TOOL     input_tensor = pad_tensor_by_size(input_tensor, pad_size)
DEBUG TOOL   File "/tmp/foundation-model-stack/fms/modules/ssm.py", line 21, in pad_tensor_by_size
DEBUG TOOL     return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)
DEBUG TOOL   File "/usr/local/lib64/python3.11/site-packages/torch/nn/functional.py", line 5096, in pad
DEBUG TOOL     return torch._C._nn.pad(input, pad, mode, value)
```

### layer_debugging mode
```
python3 deepview.py --model_type fms --model /mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --mode layer_debugging --output_file debugger.txt
```

> [!NOTE]
> The output for `layer_debugging` comes after the code has identified the crash issue and hence is printed on the screen post the crash mesaage occurs. Please do not do *Ctrl+C* after the crash and wait for deepview to exit gracefully.


#### Sample Output

```
======================================================
DEBUG TOOL update lazy handle for model.base_model.layers[0].attn, input string [1, 10, 2048], data type torch.float16
Failed layer is  model.base_model.layers[0].attn, input string [1, 10, 2048], data type torch.float16
The repro code is stored in file layer_repro_code.py
======================================================
```
