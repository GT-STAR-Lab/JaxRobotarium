"""
Generates a folder with all files necessary for Robotarium deployment.
"""
import argparse
import torch
import os
import json
import yaml
import importlib
import numpy as np
import shutil
import re

from safetensors.flax import load_file

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

def flax_to_torch(flax_state_dict, torch_state_dict):
    """
    Convert flax state dict to torch state dict

    Args:
        flax_state_dict: (Dict) dictionary of names and associated parameters
        torch_state_dict: (Dict) dictionary of names and associated parameters

    Returns:
        (Dict) matching torch_state_dict format with parameters from flax_state_dict
    """

    def _bias(param):
        return torch.from_numpy(param)

    def _dense(param):
        return torch.from_numpy(param.T)

    param_map = {}
    for name, param in flax_state_dict.items():
        param = np.array(param)
        # skip all non agent parameters
        if any('agent' in key for key in flax_state_dict.keys()) and 'agent' not in name:
            continue
        
        if 'Dense' in name:
            if 'kernel' in name:
                torch_state_dict[f'{name.split(",")[-2]}.weight'] = _dense(param)
                param_map[f'{name.split(",")[-2]}.weight'] = name
            if 'bias' in name:
                torch_state_dict[f'{name.split(",")[-2]}.bias'] = _bias(param)
                param_map[f'{name.split(",")[-2]}.bias'] = name

        if 'GRUCell' in name:
            gru_param = name.split(",")[-2]
            N = param.shape[0]
            if 'ir' in gru_param or 'iz' in gru_param or 'in' in gru_param:
                prefix = 'W'
                gate_map = {'ir': 'r', 'iz': 'z', 'in': 'h'}
            else:  # hr, hz, hn
                prefix = 'U'
                gate_map = {'hr': 'r', 'hz': 'z', 'hn': 'h'}
            
            gate = gate_map[gru_param]

            if 'kernel' in name:
                torch_state_dict[f'{name.split(",")[-3]}.{prefix}_{gate}'] = _dense(param).T
                param_map[f'{name.split(",")[-3]}.{prefix}_{gate}'] = name
            if 'bias' in name:
                # special handling for n
                if 'n' in gru_param:
                    torch_state_dict[f'{name.split(",")[-3]}.b_{"i" if prefix == "W" else "h"}h'] = _bias(param)
                    param_map[f'{name.split(",")[-3]}.b_{"i" if prefix == "W" else "h"}h'] = name
                else:
                    torch_state_dict[f'{name.split(",")[-3]}.b_{gate}'] = _bias(param)
                    param_map[f'{name.split(",")[-3]}.b_{gate}'] = name
    for key, value in param_map.items():
        print(f'{key}: {value}')
        
    return torch_state_dict

def replace_dynamic_slice_in_file(file_path):
    """Helper function to replace usages of dynamic_slice"""

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    pattern = re.compile(r"jax\.lax\.dynamic_slice\((.+?),\s*\((.*?)\),\s*\((.*?)\)\)")

    for line in lines:
        match = pattern.search(line)
        if match:
            array, starts, sizes = match.groups()
            start_vars = starts.split(", ")
            size_vars = sizes.split(", ")
            
            # Convert dynamic_slice to numpy slicing
            slices = [f"{start}:{start}+{size}" for start, size in zip(start_vars, size_vars)]
            numpy_slice = f"{array}[{', '.join(slices)}]"

            # Replace the matched dynamic slice with NumPy slicing
            line = pattern.sub(numpy_slice, line)

        new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)

def replace_jax_choice_with_np(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Match jax.random.choice(key, <rest of args>) — capturing only the rest
    pattern = re.compile(
        r"jax\.random\.choice\(\s*[^,]+,\s*(.+?)\)",
        re.DOTALL
    )

    # Replace with np.random.choice(<rest of args>)
    new_content = pattern.sub(r"np.random.choice(\1)", content)

    with open(file_path, "w") as f:
        f.write(new_content)

def comment_out_jax_random_split(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "jax.random.split" in line and not line.strip().startswith("#"):
            new_lines.append("# " + line)
        else:
            new_lines.append(line)

    with open(file_path, "w") as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment', help='folder to save deployment files')
    parser.add_argument('--config', type=str, default='config.yaml', help='configuration file for deployed scenario')
    args = parser.parse_args()

    module_dir = os.path.dirname(__file__)
    config_path = os.path.join(module_dir, args.config)

    # get experiment output dir
    output_dir = os.path.join(module_dir, 'robotarium_submissions', args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    # load actor and save as .tiff
    actor_module = importlib.import_module(f"{config.model_file[:-3]}")
    actor_class = getattr(actor_module, config.model_class)
    input_dim = config.input_dim + config.num_agents if config.preprocess_obs else config.input_dim
    actor = actor_class(input_dim, config.output_dim, config.hidden_dim)
    weights = load_file(config.model_weights)
    
    state_dict = flax_to_torch(weights, actor.state_dict())
    actor.load_state_dict(state_dict)
    torch.save(actor.state_dict(), os.path.join(output_dir, 'agent.tiff'))

    # update config and save as .npy
    config_output_path = os.path.join(output_dir, 'config.npy')
    shutil.copy(config_path, config_output_path)
    with open(config_output_path, 'r') as file:
        data = file.read()
    data = data.replace(config.model_weights, 'agent.tiff')
    data = data.replace('"save_gif": True', '"save_gif": False')
    with open(config_output_path, 'w') as file:
        file.write(data)
    
    # copy scenario and constants files
    scenario_py = config.scenario_file
    scenario_path = os.path.join(
        "/".join(module_dir.split("/")[:-1]),
        'scenarios',
        scenario_py
    )
    scenario_output_path = os.path.join(output_dir, scenario_py)
    shutil.copy(scenario_path, scenario_output_path)

    # update scenario file to not use dynamic slice
    replace_dynamic_slice_in_file(scenario_output_path)

    # update scenario file to convert jax.lax.random.choice
    replace_jax_choice_with_np(scenario_output_path)

    # comment out any key splitting logic
    comment_out_jax_random_split(scenario_output_path)

    constants_path = os.path.join(
        "/".join(module_dir.split("/")[:-1]),
        'constants.py'
    )
    constants_output_path = os.path.join(output_dir, 'constants.py')
    shutil.copy(constants_path, constants_output_path)

    # copy model, robotarium_env, and main files
    shutil.copy(os.path.join(module_dir, config.model_file), os.path.join(output_dir, config.model_file))
    shutil.copy(os.path.join(module_dir, 'robotarium_env.py'), os.path.join(output_dir, 'robotarium_env.py'))
    shutil.copy(os.path.join(module_dir, 'main.py'), os.path.join(output_dir, 'main.py'))
    