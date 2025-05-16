"""
Executed deployment script.
"""

import torch
import numpy as jnp
import os
import yaml
import json
import importlib

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
        self.__json__ = json.dumps(d, indent=4)

if __name__ == "__main__":
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "deploy":
        module_dir = ""

    # load config
    config_path = os.path.join(module_dir, 'config.npy')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    num_agents = config.pop('num_agents')
    max_steps = config.pop('max_steps')
    config = objectview(config)

    # set seed
    jnp.random.seed(config.seed)

    # load scenario
    scenario_module = importlib.import_module(f"{config.scenario_file[:-3]}")
    scenario = getattr(scenario_module, config.scenario)
    env = scenario(num_agents, max_steps, **config.__dict__)

    # load actor
    actor_module = importlib.import_module(f"{config.model_file[:-3]}")
    actor_class = getattr(actor_module, config.model_class)
    input_dim = config.input_dim + num_agents if config.preprocess_obs else config.input_dim
    actor = actor_class(input_dim, config.output_dim, config.hidden_dim)
    actor_weights = torch.load(config.model_weights)
    actor.load_state_dict(actor_weights)
   
    state = env.initial_state
    obs = env.get_obs(state)
    hs = torch.from_numpy(jnp.zeros((num_agents, config.hidden_dim))).to(torch.float32)
    one_hot_id = jnp.eye(num_agents)
    frames = []
    rewards = []
    infos = []
    for i in range(max_steps):
        # get agent action
        if config.preprocess_obs:
            obs = jnp.hstack([jnp.vstack([obs_i for obs_i in obs.values()]), one_hot_id])
        else:
            obs = jnp.vstack([obs_i for obs_i in obs.values()])
        obs = torch.from_numpy(obs).to(torch.float32)
        qvals, hs = actor(obs, hs)

        # since mappo is categorical, this works for both
        actions = {f'agent_{i}': jnp.argmax(qvals[i].detach().numpy()) for i in range(num_agents)}

        obs, state, reward, dones, info = env.step_env(None, state, actions)

        rewards.append(reward)
        infos.append(info)

        # visualize
        env.render_frame(state)
        env.visualizer.figure.canvas.draw()
        if config.save_gif:
            frame = jnp.array(env.visualizer.figure.canvas.renderer.buffer_rgba())
            frames.append(frame)

    summary_reward = 0
    for r in rewards:
        for key, value in r.items():
            summary_reward += value.mean().item()
    summary_reward = summary_reward / num_agents

    summary_info = {}
    for key, value in infos[-1].items():
        summary_info[key] = value.mean().item()

    print("="*20)
    print(f"reward: {summary_reward}")
    print(f"infos:\n{summary_info}")
    print("="*20)

    
    if config.save_gif:
        import imageio
        imageio.mimsave(f'{config.scenario.lower()}.gif', frames, duration=100, loop=0)
    
    env.robotarium.call_at_scripts_end()
