import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld,OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS

parser = argparse.ArgumentParser()
parser.add_argument("--index", default=-1, type=int)
parser.add_argument("--last_x", default=-1, type=int)
parser.add_argument("--last_y", default=-1, type=int)
args = parser.parse_args()

game_pool = ['cramped_room_tomato','cramped_room_tomato',
            'forced_coordination_tomato', 'forced_coordination_tomato',
            'soup_coordination', 'soup_coordination']

mdp_pool = []

policy_pool = []

# 创建映射关系
key_map = {
    "base.cnn.cnn.0.weight": "base.0.weight",
    "base.cnn.cnn.0.bias": "base.0.bias",
    "base.cnn.cnn.3.weight": "base.3.weight",
    "base.cnn.cnn.3.bias": "base.3.bias",
    "base.cnn.cnn.5.weight": "base.5.weight",
    "base.cnn.cnn.5.bias": "base.5.bias",
    "act.action_out.linear.weight": "action_out.weight",
    "act.action_out.linear.bias": "action_out.bias"
}

class Policy(nn.Module):
    def __init__(self, input_width,input_height, input_channels):
        super(Policy, self).__init__()
        conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(1, 1))

        # 计算卷积层输出尺寸
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_height = conv2d_size_out(input_height, 3)
        conv_width = conv2d_size_out(input_width, 3)
        linear_input_size = conv_height * conv_width * 32

        self.base = nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(linear_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.action_out = nn.Linear(64, 6)

    def forward(self, x: torch.Tensor):
        x = x.to(dtype=torch.float)
        x = x.permute((0, 3, 1, 2))  # 调整x的形状以适应网络
        x = self.base(x)
        action_out = torch.distributions.Categorical(logits=self.action_out(x))
        greedy_action = action_out.mode
        return greedy_action[0]

for i,layout_name in enumerate(game_pool):
    agent_index = i%2
    base_mdp = OvercookedGridworld.from_layout_name(layout_name)
    mdp_pool.append(base_mdp)
    
    dummy_state = base_mdp.get_standard_start_state()
    obs = mdp_pool[args.index].lossless_state_encoding(dummy_state)[agent_index]
    width = obs.shape[0]
    height = obs.shape[1]
    channels = obs.shape[2]
    # channels = 20
    policy_i = Policy(width,height,channels)
    actor_net = os.path.dirname(os.path.abspath(__file__)) + f"/{layout_name}.pth"
    policy_actor_state_dict = torch.load(str(actor_net), map_location=torch.device('cpu'))
    # 更新state_dict中的键
    new_state_dict = {key_map.get(k, k): v for k, v in policy_actor_state_dict.items()}
    policy_i.load_state_dict(new_state_dict)

     # 启用评估模式
    policy_i.eval()

    policy_pool.append(policy_i)


def my_controller(observation, action_space, is_act_continuous=False):
    if observation["new_map"]:
        args.index += 1
        args.last_x = -1
        args.last_y = -1

    state = OvercookedState.from_dict(observation)
    obs = mdp_pool[args.index].lossless_state_encoding(state)[observation["controlled_player_index"]]
    
    agent_action = []
    for i in range(len(action_space)):
        kazhu = False
        # if state.players[observation["controlled_player_index"]].position[0] == args.last_x \
        #   and state.players[observation["controlled_player_index"]].position[1] == args.last_y:
        #     kazhu = True
        
        if kazhu:  # Random
            action_ = sample_single_dim(action_space[i], is_act_continuous)
        else: # selfplay policy
            each = [0] * action_space[i].n
            idx = policy_pool[args.index](torch.from_numpy(obs[None, :]))
            
            each[idx] = 1
            action_ = each

        agent_action.append(action_)
    
    args.last_x = state.players[observation["controlled_player_index"]].position[0]
    args.last_y = state.players[observation["controlled_player_index"]].position[1]

    return agent_action


def sample_single_dim(action_space_list_each, is_act_continuous):
    each = []
    if is_act_continuous:
        each = action_space_list_each.sample()
    else:
        if action_space_list_each.__class__.__name__ == "Discrete":
            each = [0] * action_space_list_each.n
            idx = action_space_list_each.sample()
            each[idx] = 1
        elif action_space_list_each.__class__.__name__ == "MultiDiscreteParticle":
            each = []
            nvec = action_space_list_each.high - action_space_list_each.low + 1
            sample_indexes = action_space_list_each.sample()

            for i in range(len(nvec)):
                dim = nvec[i]
                new_action = [0] * dim
                index = sample_indexes[i]
                new_action[index] = 1
                each.extend(new_action)
    return each