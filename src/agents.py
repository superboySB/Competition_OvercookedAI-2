from abc import ABC, abstractmethod

import numpy as np
import torch as th


class Agent(ABC):
    """
    Base class for all agents in multi-agent environments
    """

    @abstractmethod
    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        """
        Return an action given an observation.

        :param obs: The observation to use
        :param record: Whether to record the obs, action pair (for training)
        :returns: The action to take
        """

    @abstractmethod
    def update(self, reward: float, done: bool) -> None:
        """
        Add new rewards and done information if the agent can learn.

        Each update corresponds to the most recent `get_action` (where
        `record` is True). If there are multiple calls to `update` that
        correspond to the same `get_action`, their rewards are summed up and
        the last done flag will be used.

        :param reward: The reward receieved from the previous action step
        :param done: Whether the game is done
        """
