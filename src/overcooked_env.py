import gym
import os
import torch
import numpy as np

from .multiagentenv import MultiAgentEnv
from .vectorenv import VectorMultiAgentEnv
from .vectorobservation import VectorObservation

from overcooked_ai_py.utils import load_dict_from_file
from overcooked_ai_py.mdp.actions import Action

import build.madrona_simplecooked_example_python as overcooked_python

LAYOUTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 比赛这里给的是完全一模一样的三张图: cramped_room_tomato, forced_coordination_tomato, soup_coordination
def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, "env/layouts", layout_name + ".layout"))

class OvercookedMadrona(VectorMultiAgentEnv):

    def __init__(self, layout_name, num_envs, gpu_id, debug_compile=True, use_cpu=False, use_env_cpu=False, ego_agent_idx=0, horizon=200, num_players=None):
        self.layout_name = layout_name
        self.base_layout_params = get_base_layout_params(layout_name, horizon, max_num_players=num_players)
        self.width = self.base_layout_params['width']
        self.height = self.base_layout_params['height']
        self.num_players = self.base_layout_params['num_players']
        self.size = self.width * self.height

        self.horizon = horizon

        sim = overcooked_python.SimplecookedSimulator(
            exec_mode = overcooked_python.madrona.ExecMode.CPU if use_cpu else overcooked_python.madrona.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
            **self.base_layout_params
        )

        sim_device = torch.device('cpu') if use_cpu or not torch.cuda.is_available() else torch.device('cuda')
        
        full_obs_size = self.width * self.height * (5 * self.num_players + 10)

        self.sim = sim

        self.static_dones = self.sim.done_tensor().to_torch()
        self.static_active_agents = self.sim.active_agent_tensor().to_torch().to(torch.bool)
        
        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()
        self.static_worldID = self.sim.world_id_tensor().to_torch().to(torch.long)
        self.static_agentID = self.sim.agent_id_tensor().to_torch().to(torch.long)
        self.static_locationWorldID = self.sim.location_world_id_tensor().to_torch().to(torch.long)
        self.static_locationID = self.sim.location_id_tensor().to_torch().to(torch.long)

        self.static_action_mask = torch.ones((num_envs, len(Action.ALL_ACTIONS))).to(device=sim_device, dtype=torch.bool)

        self.obs_size = full_obs_size
        self.state_size = full_obs_size
        
        self.static_scattered_active_agents = self.static_active_agents.detach().clone()
        self.static_scattered_rewards = self.static_rewards.detach().clone()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents

        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        if use_env_cpu:
            env_device = torch.device('cpu')
        else:
            env_device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        super().__init__(num_envs, device=env_device, n_players=self.num_players)

        self.static_scattered_observations = torch.empty((self.height * self.width * self.num_players, self.num_envs, 5 * self.num_players + 10), dtype=torch.int8, device=sim_device)
        self.static_scattered_observations[self.static_locationID, self.static_locationWorldID, :] = self.static_observations[:, :, :5 * self.num_players + 10]

        self.infos = [{}] * self.num_envs
        
        self.ego_ind = ego_agent_idx

        self.observation_space = self._setup_observation_space()
        self.share_observation_space = self.observation_space
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        self.n_reset()

    def _setup_observation_space(self):
        obs_shape = np.array([self.width, self.height, 5 * self.num_players + 10])
        return gym.spaces.MultiBinary(obs_shape)

    def to_torch(self, a):
        return a.to(self.device) #.detach().clone()

    def get_obs(self):
        # self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_locationID, self.static_locationWorldID, :] = self.static_observations[:, :, :5 * self.num_players + 10]

        obs0 = self.to_torch(self.static_scattered_observations[:, :, :]).reshape((self.num_players, self.height, self.width, self.num_envs, -1)).transpose(1, 3)

        obs = [VectorObservation(self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
                                 obs0[i],
                                 action_mask=self.static_action_mask)
               for i in range(self.n_players)]

        return obs

    def n_step(self, actions):
        actions_device = self.static_agentID.get_device()
        actions = actions.to(actions_device if actions_device != -1 else torch.device('cpu'))
        self.static_actions.copy_(actions[self.static_agentID, self.static_worldID, :])
        # self.static_actions.copy_(actions)
        self.sim.step()
        
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        return self.get_obs(), self.to_torch(self.static_scattered_rewards), self.to_torch(self.static_dones), self.infos

    def n_reset(self):
        return self.get_obs()

    def close(self, **kwargs):
        pass


MAX_NUM_INGREDIENTS = 3

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
}

TOMATO = "tomato"
ONION = "onion"

AIR = ' '
POT = 'P'
COUNTER = 'X'
ONION_SOURCE = 'O'
TOMATO_SOURCE = 'T'
DISH_SOURCE = 'D'
SERVING = 'S'
TERRAIN_TYPES = [AIR, POT, COUNTER, ONION_SOURCE, DISH_SOURCE, SERVING, TOMATO_SOURCE]
PLAYER_NUMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
               "!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
               "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
               "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]


def get_true_orders(orders):
    ans = [0] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
    for order in orders:
        num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
        num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
        # ans.append((num_onions, num_tomatoes))
        ans[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = 1
        # ans.append(num_onions)
        # ans.append(num_tomatoes)
    return ans


def get_base_layout_params(layout_name: str, horizon, max_num_players=None):
    if layout_name.endswith(".layout"):
        base_layout_params = load_dict_from_file(layout_name)
    else:
        base_layout_params = read_layout_dict(layout_name)
    grid = base_layout_params['grid']
    del base_layout_params['grid']

    if 'start_order_list' in base_layout_params:
        # base_layout_params['start_all_orders'] = base_layout_params['start_order_list']
        del base_layout_params['start_order_list']

    if 'num_items_for_soup' in base_layout_params:
        del base_layout_params['num_items_for_soup']

    grid = [layout_row.strip() for layout_row in grid.split('\n')]
    layout_grid = [[c for c in row] for row in grid]

    player_positions = [None] * 64
    for y, row in enumerate(layout_grid):
        for x, c in enumerate(row):
            if c in PLAYER_NUMS:
                layout_grid[y][x] = " "
                idx = PLAYER_NUMS.index(c)
                if max_num_players is None or idx < max_num_players:
                    player_positions[idx] = (x, y)
    num_players = len([x for x in player_positions if x is not None])
    player_positions = player_positions[:num_players]

    base_layout_params['height'] = len(layout_grid)
    base_layout_params['width'] = len(layout_grid[0])
    base_layout_params['terrain'] = [TERRAIN_TYPES.index(x) for row in layout_grid for x in row]
    base_layout_params['num_players'] = len(player_positions)
    # base_layout_params['start_player_positions'] = player_positions
    base_layout_params['start_player_x'] = [p[0] for p in player_positions]
    base_layout_params['start_player_y'] = [p[1] for p in player_positions]

    if 'rew_shaping_params' not in base_layout_params or base_layout_params['rew_shaping_params'] is None:
        base_layout_params['rew_shaping_params'] = BASE_REW_SHAPING_PARAMS

    base_layout_params['placement_in_pot_rew'] = base_layout_params['rew_shaping_params']['PLACEMENT_IN_POT_REW']
    base_layout_params['dish_pickup_rew'] = base_layout_params['rew_shaping_params']['DISH_PICKUP_REWARD']
    base_layout_params['soup_pickup_rew'] = base_layout_params['rew_shaping_params']['SOUP_PICKUP_REWARD']

    del base_layout_params['rew_shaping_params']

    if 'start_all_orders' not in base_layout_params or base_layout_params['start_all_orders'] is None:
        base_layout_params['start_all_orders'] = []

    if 'start_bonus_orders' not in base_layout_params or base_layout_params['start_bonus_orders'] is None:
        base_layout_params['start_bonus_orders'] = []

    original_start_all_orders = base_layout_params['start_all_orders']

    base_layout_params['start_all_orders'] = get_true_orders(original_start_all_orders)
    base_layout_params['start_bonus_orders'] = get_true_orders(base_layout_params['start_bonus_orders'])

    if 'order_bonus' not in base_layout_params:
        base_layout_params['order_bonus'] = 2

    times = [20] * ((MAX_NUM_INGREDIENTS + 1) ** 2)

    if 'onion_time' in base_layout_params and 'tomato_time' in base_layout_params:
        times = []
        for o in range(MAX_NUM_INGREDIENTS + 1):
            for t in range(MAX_NUM_INGREDIENTS + 1):
                times.append(o * base_layout_params['onion_time'] + t * base_layout_params['tomato_time'])
        del base_layout_params['onion_time']
        del base_layout_params['tomato_time']

    if 'recipe_times' in base_layout_params:
        for order, time in zip(original_start_all_orders, base_layout_params['recipe_times']):
            num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
            num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
            times[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = time

    if 'cook_time' in base_layout_params:
        times = [base_layout_params['cook_time']] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
        del base_layout_params['cook_time']

    base_layout_params['recipe_times'] = times

    values = [20] * ((MAX_NUM_INGREDIENTS + 1) ** 2)

    if 'onion_value' in base_layout_params and 'tomato_value' in base_layout_params:
        values = []
        for o in range(MAX_NUM_INGREDIENTS + 1):
            for t in range(MAX_NUM_INGREDIENTS + 1):
                values.append(o * base_layout_params['onion_value'] + t * base_layout_params['tomato_value'])
        del base_layout_params['onion_value']
        del base_layout_params['tomato_value']

    if 'recipe_values' in base_layout_params:
        for order, value in zip(original_start_all_orders, base_layout_params['recipe_values']):
            num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
            num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
            values[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = value

    if 'delivery_reward' in base_layout_params:
        # print("BRUH")
        values = [base_layout_params['delivery_reward']] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
        # print(values)
        del base_layout_params['delivery_reward']

    # for i in range(len(values)):
    #     if base_layout_params['start_bonus_orders'][i]:
    #         values[i] *= base_layout_params['order_bonus']

    #     if not base_layout_params['start_all_orders'][i]:
    #         values[i] = 0

    del base_layout_params['order_bonus']

    base_layout_params['recipe_values'] = values

    base_layout_params['horizon'] = horizon

    del base_layout_params['start_all_orders']
    del base_layout_params['start_bonus_orders']

    return base_layout_params