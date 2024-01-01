import torch
import numpy as np
import random
from src.overcooked_env import OvercookedMadrona

def generate_env(name, num_envs, layout='simple', use_env_cpu=False, use_baseline=False):
    if name == 'overcooked':
        return OvercookedMadrona(layout, num_envs, 0, debug_compile=False, use_env_cpu=use_env_cpu)
    else:
        raise Exception("Invalid environment name")

def set_seed(seed, cuda_deterministic):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
