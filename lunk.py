print("Starting LUNK(Large Universal Neural Kombiner), please wait...")
import torch, shutil, json, concurrent.futures, sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import psutil, os
import gradio as gr
blend_ratio, fp16, always_output_fp16, max_shard_size, verbose_info, force_cpu, load_sharded = 0.5, False, True, "2000MiB", True, True, True
test_prompt, test_max_length = "Test, ", 32
blend_ratio_b = 1.0 - blend_ratio
def get_cpu_threads():
    try:
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        return physical_cores, logical_cores
    except: return None
cpu_info = get_cpu_threads()
physical_cores, logical_cores = cpu_info if cpu_info else (4, 8)
def get_model_info(model):
    with torch.no_grad():
        outfo, cntent = "\n==============================\n", 0
        for name, para in model.named_parameters():
            cntent += 1
            outfo += ('{}: {}'.format(name, para.shape)) + "\n"
        outfo += ("Num Entries: " + str(cntent)) + "\n"
        outfo += ("==============================\n")
        return outfo

def merge_tensors(tensors, ratio_a, ratio_b):
    with torch.no_grad():
        for p1, p2 in tensors:
            p1.data.mul_(ratio_a).add_(p2.data * ratio_b)

def split_model_parameters(model, num_parts):
    parameters = list(model.parameters())
    part_size = (len(parameters) + num_parts - 1) // num_parts
    return [parameters[i:i + part_size] for i in range(0, len(parameters), part_size)]

def merge_models_part(part1, part2, ratio_a, ratio_b):
    merged_params = zip(part1, part2)
    valid_params = [(p1, p2) for p1, p2 in merged_params if p1.shape == p2.shape]
    merge_tensors(valid_params, ratio_a, ratio_b)

def merge_models(model1, model2, blend_ratio=0.6, num_threads=logical_cores):
    model1_params = split_model_parameters(model1, num_threads)
    model2_params = split_model_parameters(model2, num_threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for part1, part2 in zip(model1_params, model2_params):
            futures.append(executor.submit(merge_models_part, part1, part2, blend_ratio, 1.0 - blend_ratio))
        concurrent.futures.wait(futures)

def read_index_filenames(sourcedir):
    index = json.load(open(sourcedir + '/pytorch_model.bin.index.json', 'rt'))
    fl = [v for _, v in index['weight_map'].items()]
    return fl
def merge_models_and_save(model_path1, model_path2, model_path3=None):
    if not model_path1 or not model_path2:
        return "\nYou must select two directories containing models to merge and one output directory. Exiting."
    with torch.no_grad():
        if fp16:
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.float32)
        device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")

        print("Loading Model 1...")
        model1 = AutoModelForCausalLM.from_pretrained(model_path1, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
        model1 = model1.to(device)
        model1.eval()
        print("Model 1 Loaded. Dtype: " + str(model1.dtype))
        print("Loading Model 2...")
        model2 = AutoModelForCausalLM.from_pretrained(model_path2, torch_dtype='auto', low_cpu_mem_usage=True) #,torch_dtype=torch.float16
        model2 = model2.to(device)
        model2.eval()
        print("Model 2 Loaded. Dtype: " + str(model2.dtype))
        m1_info = get_model_info(model1)
        m2_info = get_model_info(model2)
        print("LUNKing models...")
        merge_models(model1, model2)
        if model_path3:
            print("Saving new model...")
            newsavedpath = model_path3+"/converted_model"
            if always_output_fp16 and not fp16:
                model1.half()
            model1.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
            print("\nSaved to: " + newsavedpath)
        else:
            print("\nOutput model was not saved as no output path was selected.")

        print("\nScript Completed.")
current_directory = os.getcwd()
def interface(input_text1, input_text2,input_text3):
    merge_models_and_save(input_text1, input_text2,input_text3)
    return "Success! Models have been LUNKed."
iface = gr.Interface(fn=interface,inputs=[gr.inputs.Dropdown(choices=os.listdir(current_directory), label="FIRST model directory"),gr.inputs.Dropdown(choices=os.listdir(current_directory), label="SECOND model directory"),gr.inputs.Dropdown(choices=os.listdir(current_directory), label="OUTPUT model directory")],outputs="text",title="LUNK(Large Universal Neural Kombiner)",description="Select some models and mash them into a new one! So long as they're the same size and architecture..",)
iface.launch()
