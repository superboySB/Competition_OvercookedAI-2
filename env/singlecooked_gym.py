# -*- coding:utf-8  -*-
import gymnasium as gym
from gymnasium.spaces import Discrete,Box
import numpy as np
import random
from typing import Optional, Union

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS

class SinglecookedAI(gym.Env):
    def __init__(self, config):
        self.n_player = 2
        self.map_name = config["map_name"]
        self.horizon = 400
        
        self.agent_mapping = [[0,1],[1,0]]
        self.player2agent_mapping = None

        self.base_mdp = OvercookedGridworld.from_layout_name(self.map_name)
        DEFAULT_ENV_PARAMS.update({"horizon":self.horizon})
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        dummy_state = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.lossless_state_encoding(dummy_state)[0]
        self.observation_space = Box(0.0, 1.0, shape=obs.shape, dtype=np.int64)
        self.action_space = Discrete(len(Action.ALL_ACTIONS))

        self.reset()
        self.env.display_states(self.env.state)

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,):
        np.random.seed(seed)
        random.seed(seed)

        # 从0和1中随机选择一个数字，作为当前决策方
        self.rl_agent_index = np.random.choice([0, 1])
        # print("rl_agent_index: ",self.rl_agent_index)
        self.random_agent_index = 1-self.rl_agent_index
        self.player2agent_mapping = self.agent_mapping[self.rl_agent_index]
        self.env.reset()
        obs = self.base_mdp.lossless_state_encoding(self.env.state)[self.rl_agent_index]
        return obs, {}

    def step(self, action):
        joint_action = []
        rl_each = [0] * self.action_space.n
        rl_each[action] = 1
        joint_action.append([np.array(rl_each)])
        random_each = [0] * self.action_space.n
        idx = self.action_space.sample()
        random_each[idx] = 1
        joint_action.append([np.array(random_each)])
        
        joint_action_decode = self.decode(joint_action)
        info_before = self.step_before_info(joint_action_decode)
        next_state, reward, map_done, info_after = self.env.step(joint_action_decode)
        obs = self.base_mdp.lossless_state_encoding(next_state)[self.rl_agent_index]

        done = truncated = map_done
        return obs, reward, done, truncated, info_before

    def step_before_info(self, env_action):
        info = {
            "env_actions": env_action,
            "player2agent_mapping": self.player2agent_mapping
        }

        return info

    def decode(self, joint_action):
        joint_action_decode = []
        joint_action_decode_tmp = []
        for nested_action in joint_action:
            if not isinstance(nested_action[0], np.ndarray):
                nested_action[0] = np.array(nested_action[0])
            joint_action_decode_tmp.append(nested_action[0].tolist().index(1))

        #swap according to agent index
        joint_action_decode_tmp2 = [joint_action_decode_tmp[i] for i in self.player2agent_mapping]


        for action_id in joint_action_decode_tmp2:
            joint_action_decode.append(Action.INDEX_TO_ACTION[action_id])

        return joint_action_decode


