

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
Clone the Deepview Repository:
```
git clone git@github.com:IBM/deepview.git 
```

A sample pod yaml has been provided in `/examples/deepview_pod.yaml`. This yaml has been tested with the latest release of DeepView and includes the following:
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

Copy the Deepview Repository into your pod
```
oc rsync deepview/ <pod-name>:/tmp/deepview/
```

Login to pod
```bash
oc rsh <pod-name> bash -l
```

## Installation
### local install
```
cd deepview
```

```shell
pip3 install -e .
```

# Usage
> [!NOTE]
> Please note that the instructions for torch_sendnn given below are temporary. We are working with torch_sendnn to get these changes incorporated into it.

First, copy `torch_sendnn` from its installation directory to `/tmp/torch_sendnn`:

If you are using the `e2e-stable` image, the installation directory of `torch_sendnn` is typically `/usr/local/lib/python3.12/site-packages/torch_sendnn`. Otherwise, you may use `python3 -m pip show torch_sendnn` to find out the installation directory.

```bash
mkdir -p /tmp/torch_sendnn && cp -r /usr/local/lib/python3.12/site-packages/torch_sendnn /tmp/torch_sendnn/
```

Replace the `/tmp/torch_sendnn/torch_sendnn/backends.py` and `/tmp/torch_sendnn/torch_sendnn/torch_sendnn.py` files with [deepview/tmp/backends.py](tmp/backends.py) and [deepview/tmp/torch_sendnn.py](tmp/torch_sendnn.py) files, respectively, given in this repository.

```bash
cp tmp/backends.py /tmp/torch_sendnn/torch_sendnn/backends.py
```

```bash
cp tmp/torch_sendnn.py /tmp/torch_sendnn/torch_sendnn/torch_sendnn.py

```

Next, set the PYTHONPATH.
```
export PYTHONPATH=/tmp/torch_sendnn:$PYTHONPATH
```

Now, run deepview as follows `deepview --help`.

```shell
usage: deepview [-h] --model_type {fms,hf} --model MODEL
                   [--mode {unsupported_op,layer_debugging} [{unsupported_op,layer_debugging} ...]]
                   [--show_details] [--generate_repro_code] --output_file OUTPUT_FILE

Script to run DeepView tool on any model.

options:
  -h, --help            show this help message and exit
  --model_type {fms,hf}
                        The type of model you want to debug - fms or hf.
  --model MODEL         Model name in HF format or model path
  --mode {unsupported_op,layer_debugging} [{unsupported_op,layer_debugging} ...]
                        Modes: [unsupported_op, layer_debugging] (Choose one or more). Default is the
                        unsupported_op mode.
  --show_details        Print stack trace and other details, valid only with unsupported_op.
  --generate_repro_code
                        Generate minimal reproducible code for unsupported operation.
  --output_file OUTPUT_FILE
                        Name of the file in which the debug tool output will be stored.
```


## Examples
A few examples demonstrating the use of unsupported_op and layer_debugging modes are shown below. A detailed list of models tested with DeepView can be found under [examples](./examples).

### unsupported_op mode

```
deepview --model_type fms --model /mnt/aiu-models-en-shared/models/ibm-ai-platform/Bamba-9B --mode unsupported_op --show_details --output_file debugger.txt
```

#### Sample Output

```
DEEPVIEW Caught error for copy_29: Operation not supported.
DEEPVIEW Data type: sen_datatype_enum.float16, Shape: [ 1 128 64 128 ]
DEEPVIEW==================================== Stack Trace ====================================
DEEPVIEW   File "/home/senuser/.local/lib/python3.12/site-packages/fms/models/bamba.py", line 431, in forward
DEEPVIEW     output, cache = self.base_model(
DEEPVIEW   File "/home/senuser/.local/lib/python3.12/site-packages/fms/models/bamba.py", line 351, in forward
DEEPVIEW     output = layer(
DEEPVIEW   File "/home/senuser/.local/lib/python3.12/site-packages/fms/models/bamba.py", line 146, in forward
DEEPVIEW     x = self.ssm(
DEEPVIEW   File "/home/senuser/.local/lib/python3.12/site-packages/fms/modules/ssm.py", line 343, in forward
DEEPVIEW     past_key_value_state.ssm_state.copy_(  # type: ignore[union-attr]
DEEPVIEW
```

### layer_debugging mode

```
deepview --model_type fms --model /mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --mode layer_debugging --output_file debugger.txt
```

> [!NOTE]
> The output for `layer_debugging` comes after the code has identified the crash issue and hence is printed on the screen post the crash mesaage occurs. Please do not do *Ctrl+C* after the crash and wait for deepview to exit gracefully.


#### Sample Output

```
DEEPVIEW========================================================================
DEEPVIEW Error running model.base_model.layers[0].attn, [1, 1, 2048], torch.float16
DEEPVIEW========================================================================
```
