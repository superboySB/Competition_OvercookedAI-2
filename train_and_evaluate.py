import os
from train.MAPPO.main_player import MainPlayer

from train.config import get_config
from pathlib import Path

from train.env_utils import generate_env

from train.partner_agents import CentralizedAgent

import torch
from torch import nn
import time
import numpy as np
import random

args = get_config().parse_args("")
args.num_env_steps = 10000000
args.episode_length = 400
args.env_name = "overcooked"
args.seed = 1
args.over_layout = "cramped_room_tomato"
args.run_dir = "sp"
args.restored = 0
args.cuda = True

args.n_rollout_threads = 128
args.ppo_epoch = 15
args.hidden_size = 64

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
print(device)

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout, use_env_cpu=(device=='cpu'), use_baseline=args.use_baseline)

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
        "train/"
        + args.hanabi_name
        + "/results/"
        + (args.run_dir)
        + "/"
        + str(args.seed)
    )
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))


## Train
config = {
    'all_args': args,
    'envs': envs,
    'device': device,
    'num_agents': 2,
    'run_dir': Path(run_dir)
}

# If you want to rerun this, remember to also run the two cells above so the environment is regenerated
# If you encounter an out of memory error, restart the runtime and start from the cell after "Restart runtime here!" above.
start = time.time()
ego = MainPlayer(config)
partner = CentralizedAgent(ego, 1)
envs.add_partner_agent(partner)
ego.run()
end = time.time()
print(f"Total time taken: {end - start} seconds")


########################################################
## Test
class Policy(nn.Module):

    def __init__(self, actor):
        super(Policy, self).__init__()

        self.base = actor.base.cnn.cnn
        self.act_layer = actor.act

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float)
        x = self.base(x.permute((0, 3, 1, 2)))
        x = self.act_layer(x, deterministic=True)
        return x[0]

run_dir = Path(run_dir)
args.model_dir = str(run_dir / 'models')

config = {
    'all_args': args,
    'envs': envs,
    'device': device,
    'num_agents': 2,
    'run_dir': run_dir
}

ego = MainPlayer(config)
ego.restore()
torch_network = Policy(ego.policy.actor)

actions = torch.zeros((2, args.n_rollout_threads, 1), dtype=int, device=device)

state1, state2 = envs.n_reset()
scores = torch.zeros(args.n_rollout_threads, device=device)
for i in range(args.episode_length):
    actions[0, :, :] = torch_network(state1.obs)
    actions[1, :, :] = torch_network(state2.obs)
    (state1, state2), reward, _, _ = envs.n_step(actions)
    scores += reward[0, :]
score_vals, counts = torch.unique(scores, return_counts=True)

# printing scores here
print({x.item() : y.item() for x, y in zip(score_vals, counts)})