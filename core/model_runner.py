# Allows running specific modules/layers in isolation with preserved context
import subprocess


def run_model(model_type, model, output_file):
    # Run the inference scripts
    if model_type == 'hf':
        command = [
            'python3', '../core/inference_hf.py', '--model_name', model,
        ]
    elif model_type == 'fms':
        command = [
            'python3', '../core/inference_fms.py', '--architecture', 'hf_pretrained',
            '--model_path', model, '--tokenizer', model, 
            '--device_type', 'aiu', '--unfuse_weights', '--compile', '--compile_dynamic',
            '--default_dtype','fp16','--fixed_prompt_length','64', '--max_new_tokens','20', 
            '--timing','per-token','--batch_size','1'
        ]

    # Show output in terminal as well as save in file
    model_output_file = "model_output.txt"
    with open(model_output_file, "w") as f:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line, end='')  
            f.write(line)        
        process.wait()

    # Pipe debug tool output to another file
    tool_output_file = output_file
    flag = True
    with open(model_output_file, "r") as infile, open(tool_output_file, "w") as outfile:
        for line in infile:
            if line.startswith("DEBUG TOOL"):
                flag = False
                outfile.write(line)
        if flag:
            outfile.write("All operators are supported.")
    ## If the code breaks, still should output 