

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

A sample pod yaml for DD2 is provided in `/examples/deepview_pod_DD2.yaml`. This yaml has been tested on an Openshift DD2 cluster and includes the following:
- Upgrade of Transformers to the latest version. This is required for the examples provided
- Installation of the Foundation Model Stack repository on a specific commit. This ensures reproducibility in the results returned by DeepView when using `--model_type=fms`
- Temporary DOOM Fix (This will be removed when fixed in base image and base image is tested with Deepview)
- Mounting of shared model PVC for the DD2 Spyre beta dev cluster. 

Please Modify the name of your pod in the following lines: 
```yaml
metadata:
  name: <pod-name>
spec:
  containers:
  - name: <pod-name>
```

and also your image pull secret:
```yaml
 imagePullSecrets:
    - name: <your-pull-secret>
```

Use the modified pod yaml to create a pod:
```bash
oc create -f modified_deepview_pod.yaml
```

Copy the Deepview Repository into your pod
```
oc rsync deepview/ <pod-name>:/home/senuser/deepview/
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
mkdir -p /home/senuser/torch_sendnn && cp -r /usr/local/lib/python3.12/site-packages/torch_sendnn /home/senuser/torch_sendnn/
```

Replace the `/home/senuser/torch_sendnn/torch_sendnn/backends.py` and `/home/senuser/torch_sendnn/torch_sendnn/torch_sendnn.py` files with [deepview/tmp/backends.py](tmp/backends.py) and [deepview/tmp/torch_sendnn.py](tmp/torch_sendnn.py) files, respectively, given in this repository.

```bash
cp tmp/backends.py /home/senuser/torch_sendnn/torch_sendnn/backends.py
```

```bash
cp tmp/torch_sendnn.py /home/senuser/torch_sendnn/torch_sendnn/torch_sendnn.py

```

Next, set the PYTHONPATH.
```
export PYTHONPATH=/home/senuser/torch_sendnn:$PYTHONPATH
```
Now, run deepview as follows :
`deepview --help`

```shell
usage: deepview [-h] --model_type {fms,hf} --model MODEL
                   [--mode {unsupported_op,layer_debugging,aiu_input_capture,layer_io_divergence} [{unsupported_op,layer_debugging,aiu_input_capture,layer_io_divergence} ...]]
                   [--show_details] [--generate_repro_code] --output_file OUTPUT_FILE --layer_inputs_file LAYER_INPUTS_FILE

Script to run DeepView tool on any model.

options:
  -h, --help            show this help message and exit
  --model_type {fms,hf}
                        The type of model you want to debug - fms or hf.
  --model MODEL         Model name in HF format or model path
  --mode {unsupported_op,layer_debugging,aiu_input_capture,layer_io_divergence} 
                        Modes: [unsupported_op, layer_debugging, aiu_input_capture, layer_io_divergence] (Choose ONLY one). Default is the
                        unsupported_op mode.
  --show_details        Print stack trace and other details, valid only with unsupported_op.
  --generate_repro_code
                        Generate minimal reproducible code for unsupported operation.
  --output_file OUTPUT_FILE
                        Name of the file in which the debug tool output will be stored.
  --layer_inputs_file LAYER_INPUTS_FILE
                        Name of the file in which AIU layer inputs are stored.
```

## Examples
A few examples demonstrating the use of unsupported_op, layer_debugging, and layer_io_divergence modes are shown below. A detailed list of models tested with DeepView can be found under [examples](./examples).

### unsupported_op mode
```
deepview --model_type fms --model ibm-ai-platform/Bamba-9B-v1 --mode unsupported_op --show_details --output_file debugger.txt
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
deepview --model_type fms --model ibm-granite/granite-3.2-2b-instruct --mode layer_debugging --output_file debugger.txt
```

> [!NOTE]
> The output for `layer_debugging` comes after the code has identified the crash issue and hence is printed on the screen post the crash mesaage occurs. Please do not do *Ctrl+C* after the crash and wait for deepview to exit gracefully.


#### Sample Output

```
DEEPVIEW========================================================================
DEEPVIEW Error running model.base_model.layers[0].attn, [1, 1, 2048], torch.float16
DEEPVIEW========================================================================
```


### layer_io_divergence mode
```
deepview --model_type fms --model ibm-granite/granite-3.2-8b-instruct --mode layer_io_divergence
```

> [!NOTE]
> The `layer_io_divergence` mode assumes that the thresholds from GPU run is already present in the path pointed by `DEEPVIEW_THRESHOLDS_FOLDERPATH` env variable (specified in the Deepview pod yaml).


#### Sample Output

```
DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.embedding.
DEEPVIEW Metric: abs_diff. Observed Value = 0.0. Threshold = 2.582696413198807e-07.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0009765625. Threshold = 1.0000495910644531.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0009765625. Threshold = 1.0000495910644531.
DEEPVIEW Threshold test passed for model.base_model.embedding.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[0].ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.39794921875. Threshold = 0.7421883531266595.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 0.99951171875. Threshold = 1.000023373582745.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 0.99951171875. Threshold = 1.000023373582745.
DEEPVIEW Threshold test passed for model.base_model.layers[0].ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[0].ff_ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.25146484375. Threshold = 0.5086808401930663.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 1.0000249248155406.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 1.0000249248155406.
DEEPVIEW Threshold test passed for model.base_model.layers[0].ff_ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[0].ff_sub_layer.
DEEPVIEW Metric: abs_diff. Observed Value = 0.19580078125. Threshold = 0.32078647954976347.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 0.9999915364626292.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 0.9999915364626292.
DEEPVIEW Threshold test passed for model.base_model.layers[0].ff_sub_layer.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[1].ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.392822265625. Threshold = 1.377269982083503.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 1.0000008151337907.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 1.0000008151337907.
DEEPVIEW Threshold test passed for model.base_model.layers[1].ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[1].ff_ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.359130859375. Threshold = 0.773005896516742.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 0.9999926803268004.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 0.9999926803268004.
DEEPVIEW Threshold test passed for model.base_model.layers[1].ff_ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[1].ff_sub_layer.
DEEPVIEW Metric: abs_diff. Observed Value = 0.2418212890625. Threshold = 0.3038148235646063.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 0.99951171875. Threshold = 1.0000109166315156.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 0.99951171875. Threshold = 1.0000109166315156.
DEEPVIEW Threshold test passed for model.base_model.layers[1].ff_sub_layer.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[2].ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.59375. Threshold = 0.889103195155278.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 0.99951171875. Threshold = 1.00001675180263.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 0.99951171875. Threshold = 1.00001675180263.
DEEPVIEW Threshold test passed for model.base_model.layers[2].ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[2].ff_ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.434814453125. Threshold = 0.5860415770837124.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 0.999989212775717.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 0.999989212775717.
DEEPVIEW Threshold test passed for model.base_model.layers[2].ff_ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[2].ff_sub_layer.
DEEPVIEW Metric: abs_diff. Observed Value = 0.27685546875. Threshold = 0.49616399385390714.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 0.9990234375. Threshold = 0.9999972087772269.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 0.9990234375. Threshold = 0.9999972087772269.
DEEPVIEW Threshold test passed for model.base_model.layers[2].ff_sub_layer.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[3].ln.
DEEPVIEW Metric: abs_diff. Observed Value = 1.3837890625. Threshold = 1.9169280779069073.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 0.99951171875. Threshold = 1.0000057428382163.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 0.99951171875. Threshold = 1.0000057428382163.
DEEPVIEW Threshold test passed for model.base_model.layers[3].ln.
DEEPVIEW========================================================================

DEEPVIEW========================================================================
DEEPVIEW Layer is model.base_model.layers[3].ff_ln.
DEEPVIEW Metric: abs_diff. Observed Value = 0.6962890625. Threshold = 0.5865474108397013.
DEEPVIEW Metric: cos_sim_avg. Observed Value = 1.0. Threshold = 0.9999819579003733.
DEEPVIEW Metric: cos_sim_mean. Observed Value = 1.0. Threshold = 0.9999819579003733.
DEEPVIEW Threshold test failed for model.base_model.layers[3].ff_ln.
DEEPVIEW========================================================================

DeepView run completed
```
