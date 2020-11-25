import gym
import torch
import numpy as np
from torch import nn
from numba import njit
from abc import ABC, abstractmethod
from typing import Any, List, Union, Mapping, Optional, Callable

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class BasePolicy(ABC, nn.Module):
    def __init__(
        self,
        observation_space: gym.Space = None,
        action_space: gym.Space = None
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.agent_id = 0
        self.updating = False
        self._compile()

    @abstractmethod
    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        pass

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.
        Used in :meth:`update`. Check out :ref:`process_fn` for more
        information.
        """
        return batch

    @abstractmethod
    def learn(
        self, batch: Batch, **kwargs: Any
    ) -> Mapping[str, Union[float, List[float]]]:
        """Update policy with a given batch of data.
        :return: A dict which includes loss and its corresponding label.
        .. note::
            In order to distinguish the collecting state, updating state and
            testing state, you can check the policy state by ``self.training``
            and ``self.updating``. Please refer to :ref:`policy_state` for more
            detailed explanation.
        .. warning::
            If you use ``torch.distributions.Normal`` and
            ``torch.distributions.Categorical`` to calculate the log_prob,
            please be careful about the shape: Categorical distribution gives
            "[batch_size]" shape while Normal distribution gives "[batch_size,
            1]" shape. The auto-broadcasting of numerical operation with torch
            tensors will amplify this error.
        """
        pass

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.
        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        if hasattr(buffer, "update_weight") and hasattr(batch, "weight"):
            buffer.update_weight(indice, batch.weight)

    def update(
        self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any
    ) -> Mapping[str, Union[float, List[float]]]:
        """Update the policy network and replay buffer.
        It includes 3 function steps: process_fn, learn, and post_process_fn.
        In addition, this function will change the value of ``self.updating``:
        it will be False before this function and will be True when executing
        :meth:`update`. Please refer to :ref:`policy_state` for more detailed
        explanation.
        :param int sample_size: 0 means it will extract all the data from the
            buffer, otherwise it will sample a batch with given sample_size.
        :param ReplayBuffer buffer: the corresponding replay buffer.
        """
        if buffer is None:
            return {}
        batch, indice = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indice)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indice)
        self.updating = False
        return result