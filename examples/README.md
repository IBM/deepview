# Models tested with DeepView with Release v0.1.0



| Model                          | Command                                                                                                                                                                                                         | Unsupported Ops                                                        | 
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | 
| Bamba 9B (FMS)                 | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/ibm-ai-platform/Bamba-9B --mode unsupported_op --output_file debugger.txt --show_details --generate_repro_code    | constant_pad_nd<br /> copy<br /> roll<br />select_scatter <br /> slice_scatter<br /> softplus             | 
| granite-3.2-2b-instruct (FMS)  | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --mode unsupported_op --output_file debugger.txt --show_details                        | No unsupported operations detected.                                    | 
| Mistral-7B-Instruct-v0.3 (FMS) | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/hf/Mistral-7B-Instruct-v0.3 --mode unsupported_op  --output_file debugger.txt --show_details --generate_repro_code | No unsupported operations detected.                                    | 
| Mistral-7B-Instruct-v0.3 (HF)  | deepview --model_type hf --model mistralai/Mistral-7B-Instruct-v0.3 --mode unsupported_op  --output_file debugger.txt --show_details --generate_repro_code                            | copy<br />cos<br />gt<br />sin<br />slice_scatter                                              | 
| Bamba-9B-v2 (HF)               | deepview --model_type hf --model ibm-ai-platform/Bamba-9B-v2 --mode unsupported_op  --output_file debugger.txt --show_details --generate_repro_code                                   | constant_pad_nd<br />copy<br />cos<br />gt<br />roll<br />select_scatter<br />sin<br />slice_scatter<br />softplus<br />triu | 
| granite-3.3-2b-instruct (HF)   | deepview --model_type hf --model ibm-granite/granite-3.3-2b-instruct --mode unsupported_op  --output_file debugger.txt --show_details --generate_repro_code                           | copy<br />cos<br />gt<br />sin<br />slice_scatter<br />triu                                          | 
| all-mpnet-base-v2(HF)          | deepview --model_type hf --model sentence-transformers/all-mpnet-base-v2 --mode unsupported_op  --output_file debugger.txt --show_details --generate_repro_code                       | full_like<br />lt<br />minimum                                                     | 




| Model                          | Command                                                                                                                                                                                                         |  Offending Layer                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |  -------------------------------- |
| Bamba 9B (FMS)                 | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/ibm-ai-platform/Bamba-9B --mode  layer_debugging --output_file debugger.txt --generate_repro_code    | model.base_model.layers[0].ssm   |
| granite-3.2-2b-instruct (FMS)  | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/hf/granite-3.2-2b-instruct --mode layer_debugging --output_file debugger.txt                        | model.base_model.layers[0].attn  |
| Mistral-7B-Instruct-v0.3 (FMS) | deepview --model_type fms --model /mnt/aiu-models-en-shared/models/hf/Mistral-7B-Instruct-v0.3 --mode layer_debugging --output_file debugger.txt --generate_repro_code | No model layer has failed        |
| Mistral-7B-Instruct-v0.3 (HF)  | deepview --model_type hf --model mistralai/Mistral-7B-Instruct-v0.3 --mode layer_debugging --output_file debugger.txt --generate_repro_code                            | model.model.rotary_emb           |
| Bamba-9B-v2 (HF)               | deepview --model_type hf --model ibm-ai-platform/Bamba-9B-v2 --mode layer_debugging --output_file debugger.txt --generate_repro_code                                   |  model.model.rotary_emb           |
| granite-3.3-2b-instruct (HF)   | deepview --model_type hf --model ibm-granite/granite-3.3-2b-instruct --mode layer_debugging --output_file debugger.txt --generate_repro_code                           | model.model.rotary_emb           |
| all-mpnet-base-v2(HF)          | deepview --model_type hf --model sentence-transformers/all-mpnet-base-v2 --mode layer_debugging --output_file debugger.txt --generate_repro_code                       | model.encoder.layer[0].attention |
