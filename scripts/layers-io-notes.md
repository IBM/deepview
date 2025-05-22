# Layers Input and Output (io) Demo

# Checkout the io branch 
```shell
git clone git@github.com:IBM/deepview.git
cd deepview
git fetch origin
git checkout re-structure
```

# Saving the layer by layer IOs for CPU run 
```shell
cd scripts

python3 fms_save_ios.py --architecture=hf_pretrained --model_path=/mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --tokenizer=/mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --device_type=aiu --unfuse_weights --compile_dynamic --compile --default_dtype=fp16 --fixed_prompt_length=64  --max_new_tokens=2 --timing=per-token --batch_size=1

```
This should start saving the inputs and outputs of each layer/sub-layer under a directoty `saved_artificats`. At completion you should see a message as following:

```shell
Layer stack pickled successfully to: saved_artifacts/model_layers_stack.pkl
End: fms_layers saved
```
You can explore the `saved_artifcats` dir by running tree like `tree -d saved_artifacts`  and get:
```shell
saved_artifacts
├── model_0
├── model_1
├── model_base_model_0
├── model_base_model_1
├── model_base_model_dec_norm_0
├── model_base_model_dec_norm_1
├── model_base_model_embedding_0
├── model_base_model_embedding_1
├── model_base_model_layers_0_attn_dense_0
├── model_base_model_layers_0_attn_dense_1
├── model_base_model_layers_0_attn_in_proj_0
├── model_base_model_layers_0_attn_in_proj_1
├── model_base_model_layers_0_attn_in_proj_key_0
├── model_base_model_layers_0_attn_in_proj_key_1
├── model_base_model_layers_0_attn_in_proj_query_0
├── model_base_model_layers_0_attn_in_proj_query_1
├── model_base_model_layers_0_attn_in_proj_value_0
├── model_base_model_layers_0_attn_in_proj_value_1
├── model_base_model_layers_0_ff_ln_0
├── model_base_model_layers_0_ff_ln_1
├── model_base_model_layers_0_ff_sub_layer_0
├── model_base_model_layers_0_ff_sub_layer_1
├── model_base_model_layers_0_ff_sub_layer_a_0
├── model_base_model_layers_0_ff_sub_layer_a_1
├── model_base_model_layers_0_ff_sub_layer_w1_0
├── model_base_model_layers_0_ff_sub_layer_w1_1
├── model_base_model_layers_0_ff_sub_layer_w2_0
├── model_base_model_layers_0_ff_sub_layer_w2_1
├── model_base_model_layers_0_ff_sub_layer_wg_0
├── model_base_model_layers_0_ff_sub_layer_wg_1
├── model_base_model_layers_0_ln_0
├── model_base_model_layers_0_ln_1
├── model_head_0
└── model_head_1
```


# Compiling and running layers on AIU 
Run the test script as `python3 test-layer-io-with-aiu.py`, this will load the model weights and list the avialable layers for testing.
The script uses pycony to open intractive console to allow manual testing of the layers.  For example to run `model.base_model.layers[0].ln` run 

```python 
compile_and_run_layer(model,'saved_artifacts/model_base_model_layers_0_attn_in_proj_query_1')
```
I noticed that the first run, produces a large varianace, while subsequent runs produce stable very low differnce. 

At this point you can completely quit, by typing `quit()` or hit `ctrl-D` to let the auto run loop testing of the saved inputs and output of each saved layers.

