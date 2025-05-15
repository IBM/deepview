
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
DeepView is a modular debugging and diagnostics toolkit designed to streamline the model enablement workflow for the Spyre AIU accelerator. It helps bridge the gap between model developers and compiler/runtime engineers by enabling fast, precise, and reproducible identification of issues during model lowering and execution on the AIU.
## Current Capabilities
* Unsupported Op Detection: Automatically detects unsupported operators and captures detailed metadata including
	- Operator name
  - Input shapes and data types
  - (Optional) Precise stack traces mapping to source code
* Granular Debugging: Identifies offending layers and modules with fine-grained context, helping isolate
	- Input characteristics triggering the issue
  - Specific locations in the model's forward path
*	Minimal Reproduction Scripts: Generates self-contained, minimal scripts to reproduce failures and aid debugging workflows.
* Compiler and Runtime Integration: Hooks into the model compilation pipeline to trace issues early and often.
## Planned / Future Capabilities (draft)
* Intermediate Tensor Comparison: Layer-wise comparison of outputs between CPU/GPU and AIU to identify numerical mismatches or accuracy regressions.
*	Performance Diagnostics: Integration with runtime metrics (FLOPs, memory usage, latency) to pinpoint bottlenecks and inefficiencies.
*	vLLM Testing: Support for running and validating model behavior on VLLM.
*	Visualization Tools: Graph-based interfaces for analyzing PyTorch FX graphs, unsupported paths, and fallback decisions.


# Environment & Installation
## Environment Setup
A Sample pod yaml has been provided in `/examples/deepview_pod.yaml`. This is the yaml has been tested with the latest release of DeepView and includes the following:
- Upgrade of Transformers to the latest version. This is required for the examples provided
- Installation of the Foundation Model Stack repository on a specific commit. This ensures reproducibility in the results returned by DeepView when using `--model_type=fms`
- Setting of DeepTools 2.0 Environment Variables for DD1 Hardware

Please Modify the name of your pod in the following lines: 
```yaml
metadata:
  name: <pod-name>
spec:
  containers:
  - name: <pod-name>
```

Use the modified pod yaml to create a pod:
```bash
oc create -f modified_deepview_pod.yaml
```

Login to pod
```bash
oc rsh <pod-name> bash -l
```

## Installation
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

```bash
cp /usr/local/lib/python3.12/site-packages/torch_sendnn/ /tmp/torch_sendnn
```

Replace the `/tmp/torch_sendnn/backends.py` and `/tmp/torch_sendnn/torch_sendnn.py` files with [deepview/core/tmp/backends.py](/core/tmp/backends.py) and [deepview/core/tmp/torch_sendnn.py](/core/tmp/torch_sendnn.py) files, respectively, given in this repository.

```bash
cp deepview/core/tmp/backends.py /tmp/torch_sendnn/backends.py
```

```bash
cp deepview/core/tmp/torch_sendnn.py /tmp/torch_sendnn/torch_sendnn.py

```

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
