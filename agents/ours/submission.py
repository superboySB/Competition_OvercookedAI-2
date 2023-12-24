import argparse
import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from singleagent import SingleRLAgent

parser = argparse.ArgumentParser()
parser.add_argument("-obs_space", default=4, type=int)
parser.add_argument("-action_space", default=2, type=int)
parser.add_argument("-hidden_size", default=64, type=int)
parser.add_argument("-algo", default="dqn", type=str)
parser.add_argument("-network", default="critic", type=str)
parser.add_argument("-n_player", default=1, type=int)
args = parser.parse_args()

agent = SingleRLAgent(args)
critic_net = os.path.dirname(os.path.abspath(__file__)) + "/critic_200.pth"
agent.load(critic_net)

sys.path.pop(-1)  # just for safety

# def my_controller(obs_list, action_space_list, obs_space_list):
#     action = agent.choose_action_to_env(obs_list)
#     return action

def my_controller(observation, action_space, is_act_continuous=False):  # TODO：切换policy
    agent_action = []
    for i in range(len(action_space)):
        action_ = sample_single_dim(action_space[i], is_act_continuous)
        agent_action.append(action_)
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