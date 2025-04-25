# aiu-model-debugger
DeepView is a diagnostic tool to accelerate the model enablement journey on AIU. It identifies issues during model lowering and pinpoints root causes with minimal granularity.

In the current version, DeepView identifies unsupported ops with detailed metadata (such as input shapes and dtypes). If enabled, it also provides precise stack traces to locate the unsupported op in the model source code. This empowers quick root-cause analysis during model enablement journey.

The upcoming version of DeepView will expand its capabilities to automatically generating minimal reproducible scripts for unsupported ops.
It will also go beyond unsupported ops to surface deeper issues, such as internal compiler assertion failures, and precisely identify the offending layer within the transformer blocks.

# Usage
```
usage: deepview.py [-h] --model_type {fms,hf} --model MODEL --mode {unsupported_op} [{unsupported_op} ...] [--show_details] --output_file OUTPUT_FILE

Script to run DeepView tool on any model.

options:
  -h, --help            show this help message and exit
  --model_type {fms,hf}
                        The type of model you want to debug - fms or hf.
  --model MODEL         Model name in HF format or model path
  --mode {unsupported_op} [{unsupported_op} ...]
                        Modes: [unsupported_op] (Choose one or more)
  --show_details        Print stack trace and other details, valid only with unsupported_op.
  --output_file OUTPUT_FILE
                        Print stack trace and other details, valid only with unsupported_op.
```

## Example
```
python3 deepview.py --model_type fms --model /mnt/aiu-models-en-shared/models/ibm-ai-platform/Bamba-9B --mode unsupported_op --show_details --output_file debugger.txt
```

## Sample Output

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

# Initial directory structure 

```shell
aiu-model-debugger
├── LICENSE
├── README.md
├── deepview.py
├── configs
│   ├── README.md
│   └── debug_profiles.json
├── core
│   ├── README.md
│   ├── correctness.py
│   ├── fx_graph_analyzer.py
│   ├── hook_monitor.py
│   ├── model_runner.py
│   ├── op_mapper.py
│   ├── unsupported_db.py
│   ├── backends.py (will be used from torch_sendnn in future)
│   └── inference_fms.py (will be removed in future)
├── examples
│   └── README.md
├── scripts
│   ├── README.md
│   └── isolate_layer.py
├── setup.py
├── tests
│   ├── README.md
│   └── test_cases
└── utils
    ├── README.md
    └── logger.py
```
