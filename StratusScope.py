import os
import subprocess
import json
import re
import torch
from transformers import AutoModel, AutoConfig, logging
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from colorama import init, Fore, Style

logging.set_verbosity_warning()
logging.set_verbosity_error()

def select_folder():
    Tk().withdraw()
    folder = filedialog.askdirectory()
    return folder

def clear_console():
    if os.name == "nt":  # For Windows
        subprocess.call("cls", shell=True)
    else:  # For Linux and macOS
        subprocess.call("clear", shell=True)

def get_layer_number(name):
    # Use regular expression to extract layer number
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else None

def load_weight_index(folder):
    """Load the index file to get the weight to shard mapping."""
    with open(os.path.join(folder, 'pytorch_model.bin.index.json'), 'r') as f:
        index = json.load(f)
    return index['weight_map']

def load_sharded_layer(folder, target_layer, weight_index):
    layer_state_dict = {}

    # Extract all weights for the target layer using a list comprehension
    relevant_weights = {k: v for k, v in weight_index.items() if f"model.layers.{target_layer}." in k}

    # Find unique shards that contain these weights
    unique_shards = set(relevant_weights.values())

    # Now, for each unique shard, load it once and extract all weights from it
    for shard_name in unique_shards:
        shard = torch.load(os.path.join(folder, shard_name), map_location=torch.device('cuda'))
        for weight_name in relevant_weights:
            if relevant_weights[weight_name] == shard_name:
                layer_state_dict[weight_name] = shard[weight_name]

    return layer_state_dict

def get_total_layers(model_folder):
    files = os.listdir(model_folder)
    model_files = sorted([f for f in files if f.startswith('pytorch_model-') and f.endswith('.bin')])

    all_layers = set()

    for model_file in model_files:
        shard = torch.load(os.path.join(model_folder, model_file), map_location=torch.device('cuda'))
        for name in shard.keys():
            layer_number = get_layer_number(name)
            all_layers.add(layer_number)

    return len(all_layers)

# https://pytorch.org/docs/stable/tensors.html

def compare_layers(model1_folder, model2_folder):
    layer_diffs = []
    newline = '\n'
    num_layers = get_total_layers(model1_folder) - 1
    print(f"Torch Version: {torch.__version__}")
    print(f"Total Layers Found: {num_layers}{newline}")

    # Load the weight indices
    model1_weight_index = load_weight_index(model1_folder)
    model2_weight_index = load_weight_index(model2_folder)

    for layer_number in range(num_layers):
        layer_diff = 0

        model1_layer = load_sharded_layer(model1_folder, layer_number, model1_weight_index)
        model2_layer = load_sharded_layer(model2_folder, layer_number, model2_weight_index)

        for n1, p1 in model1_layer.items():
            p2 = model2_layer[n1]

            print(f"{newline}{Fore.YELLOW}--------Found Tensor Pair--------{newline}")
            print(f"p1 = {p1}")
            print(f"p2 = {p2}")
            print(f"{newline}{Fore.GREEN}--------Casting p1 & p2 tensor pair to float32--------{newline}")
            p1 = p1.detach().to(torch.float32)
            print(f"p1 = {p1}")
            p2 = p2.detach().to(torch.float32)
            print(f"p2 = {p2}")
            
            if not (torch.isinf(p1).any() or torch.isinf(p2).any()):
                diff = torch.abs(p1 - p2).sum().item()
                layer_diff += diff

        print(f"{newline}{Fore.CYAN}----------- Layer {layer_number}: Aggregate Difference = {layer_diff} -----------{Style.RESET_ALL}{newline}")
        layer_diffs.append(layer_diff)

    return layer_diffs

def plot_layer_diff(layer_diffs, model1_name, model2_name):
    plt.figure(figsize=(20, 6))
    num_layers = len(layer_diffs)
    layer_indices = range(num_layers)
    plt.bar(layer_indices, layer_diffs)
    plt.xticks(layer_indices)
    plt.xlabel('Layer')
    plt.ylabel('Difference')
    plt.title(f"{model1_name} vs {model2_name} Layer Difference")
    plt.ylim(bottom=0)
    print("Script completed, close graph to unload models and return to commandline.")
    plt.show()

def main():
    print("Select model1 folder:")
    model1_folder = select_folder()
    model1_name = os.path.basename(model1_folder)
    print("Select model2 folder:")
    model2_folder = select_folder()
    model2_name = os.path.basename(model2_folder)

    print("Examining Models...")
    clear_console()
    layer_diffs = compare_layers(model1_folder, model2_folder)

    plot_layer_diff(layer_diffs, model1_name, model2_name)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    main()
