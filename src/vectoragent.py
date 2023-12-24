from abc import ABC, abstractmethod
from .vectorobservation import VectorObservation

import torch
import torch.nn as nn

from typing import Optional

class VectorAgent(ABC):
    @abstractmethod
    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        """
        Return an action given an observation.
        :param obs: The observation to use
        :param record: Whether to record the obs, action pair (for training)
        :returns: The action to take
        """

    @abstractmethod
    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Add new rewards and done information if the agent can learn.
        Each update corresponds to the most recent `get_action` (where
        `record` is True). If there are multiple calls to `update` that
        correspond to the same `get_action`, their rewards are summed up and
        the last done flag will be used.
        :param reward: The rewards receieved from the previous action step
        :param done: Whether the game is done
        """


class RandomVectorAgent(VectorAgent):
    def __init__(self, sampler):
        self.sampler = sampler
        
    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        return self.sampler()

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        return


    
