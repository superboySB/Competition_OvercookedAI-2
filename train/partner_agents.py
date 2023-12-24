from src.vectoragent import VectorAgent

from train.MAPPO.main_player import MainPlayer

import torch


class CentralizedAgent(VectorAgent):
    def __init__(self, cent_player: MainPlayer, player_id: int, policy=None):
        self.cent_player = cent_player
        self.player_id = player_id
        if policy is None:
            self.actor = self.cent_player.trainer.policy.actor
        else:
            self.actor = policy

    def get_action(self, obs, record=True):
        available_actions = obs.action_mask
        share_obs = obs.state
        # choose = obs.active
        obs = obs.obs

        self.cent_player.trainer.prep_rollout()

        (action, action_log_prob, rnn_state) = self.actor(
            obs,
            self.cent_player.turn_rnn_states[:, self.player_id],
            self.cent_player.turn_masks[:, self.player_id],
            available_actions,
        )
        critic = self.cent_player.trainer.policy.critic
        (value, rnn_state_critic) = critic(
            share_obs,
            self.cent_player.turn_rnn_states_critic[:, self.player_id],
            self.cent_player.turn_masks[:, self.player_id],
        )

        if record:
            self.cent_player.turn_obs[:, self.player_id] = obs
            self.cent_player.turn_share_obs[:, self.player_id] = share_obs
            self.cent_player.turn_available_actions[
                :, self.player_id
            ] = available_actions
            self.cent_player.turn_values[:, self.player_id] = value
            self.cent_player.turn_actions[:, self.player_id] = action
            self.cent_player.turn_action_log_probs[:, self.player_id] = action_log_prob
            self.cent_player.turn_rnn_states[:, self.player_id] = rnn_state
            self.cent_player.turn_rnn_states_critic[:, self.player_id] = rnn_state_critic
            self.cent_player.turn_rewards[:, self.player_id] = 0
            self.cent_player.turn_active_masks[:, self.player_id] = 1

        return action

    def update(self, rewards, dones):
        rewards = rewards
        dones = dones.to(torch.bool)
        # print(dones)
        # print(rewards, self.cent_player.turn_rewards[:, 1])
        self.cent_player.turn_rewards[:, self.player_id] += rewards[:, None]

        self.cent_player.turn_masks[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states_critic[dones, self.player_id] = 0

        self.cent_player.turn_masks[~dones, self.player_id] = 1

