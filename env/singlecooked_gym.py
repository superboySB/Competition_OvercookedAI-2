# -*- coding:utf-8  -*-
import gymnasium as gym
from gymnasium.spaces import Discrete,Box
import numpy as np
import random
from typing import Optional, Union

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, DEFAULT_ENV_PARAMS
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from ray.rllib.env.env_context import EnvContext

class SinglecookedAI(gym.Env):
    def __init__(self, config: EnvContext):
        self.n_player = 2
        self.map_name = config["map_name"]
        self.use_phi = config["use_phi"]
        self._initial_reward_shaping_factor = 1.0
        self.reward_shaping_factor = 1.0
        self.reward_shaping_horizon = config["reward_shaping_horizon"]
        self.horizon = 400
        
        self.agent_mapping = [[0,1],[1,0]]
        self.player2agent_mapping = None

        self.base_mdp = OvercookedGridworld.from_layout_name(self.map_name)
        DEFAULT_ENV_PARAMS.update({"horizon":self.horizon})
        self.env = OvercookedEnv.from_mdp(self.base_mdp, **DEFAULT_ENV_PARAMS)
        dummy_state = self.base_mdp.get_standard_start_state()
        obs = self.base_mdp.lossless_state_encoding(dummy_state)[0]
        self.observation_space = Box(low=-100.0, high=100.0, shape=obs.shape, dtype=np.float32)
        self.action_space = Discrete(len(Action.ALL_ACTIONS))

        self.reset()
        self.env.display_states(self.env.state)

    def set_reward_shaping_factor(self, factor):
        self.reward_shaping_factor = factor
        
    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None):
        np.random.seed(seed)
        random.seed(seed)

        # 从0和1中随机选择一个数字，作为当前决策方
        self.rl_agent_index = np.random.choice([0, 1])
        # print("rl_agent_index: ",self.rl_agent_index)
        self.random_agent_index = 1-self.rl_agent_index
        self.player2agent_mapping = self.agent_mapping[self.rl_agent_index]
        self.env.reset()
        obs = self.base_mdp.lossless_state_encoding(self.env.state)[self.rl_agent_index]
        return obs.astype(np.float32), {}
    
    def anneal_reward_shaping_factor(self, timesteps):
        """
        Set the current reward shaping factor such that we anneal linearly until self.reward_shaping_horizon
        timesteps, given that we are currently at timestep "timesteps"
        """
        new_factor = self._anneal(
            self._initial_reward_shaping_factor,
            timesteps,
            self.reward_shaping_horizon,
        )
        self.set_reward_shaping_factor(new_factor)

    def _anneal(self, start_v, curr_t, end_t, end_v=0, start_t=0):
        if end_t == 0:
            # No annealing if horizon is zero
            return start_v
        else:
            off_t = curr_t - start_t
            # Calculate the new value based on linear annealing formula
            fraction = max(1 - float(off_t) / (end_t - start_t), 0)
            return fraction * start_v + (1 - fraction) * end_v
          
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
        next_state, sparse_reward, map_done, info_after = self.env.step(joint_action_decode,display_phi=self.use_phi)
        obs = self.base_mdp.lossless_state_encoding(next_state)[self.rl_agent_index]
        if self.use_phi:
            potential = info_after["phi_s_prime"] - info_after["phi_s"]
            dense_reward = (potential, potential)
        else:
            dense_reward = info_after["shaped_r_by_agent"]
        if self.rl_agent_index == 0:
            shaped_reward = (
                sparse_reward + self.reward_shaping_factor * dense_reward[0]
            )
        else:
            shaped_reward = (
                sparse_reward + self.reward_shaping_factor * dense_reward[1]
            )
        done = truncated = map_done
        return obs.astype(np.float32), shaped_reward, done, truncated, info_before

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


class TrainingCallbacks(DefaultCallbacks):
    # Executes at the end of a call to Trainer.train, we'll update environment params (like annealing shaped rewards)
    def on_train_result(self, algorithm, result, **kwargs):
        # Anneal the reward shaping coefficient based on environment paremeters and current timestep
        timestep = result["timesteps_total"]
        algorithm.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.anneal_reward_shaping_factor(timestep)
            )
        )
    