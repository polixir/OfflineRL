import torch
import numpy as np
from torch import nn
from batchrl.utils.net.common import BasePolicy, MLP
from typing import Any, Dict, Tuple, Union, Optional, Sequence

from batchrl.utils.data import to_torch, to_torch_as, to_numpy


SIGMA_MIN = -20
SIGMA_MAX = 2


class Actor(nn.Module, BasePolicy):
    """Simple actor network with MLP.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        max_action: float = 1.0,
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._max = max_action
        
    def policy_infer(self, obs):
        return self(obs)[0]
        
    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        """Mapping: s -> logits -> action."""
        logits, h = self.preprocess(s, state)
        logits = self._max * torch.tanh(self.last(logits))
        return logits, h


class Critic(nn.Module):
    """Simple critic network with MLP.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, 1)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = s.flatten(1)
        if a is not None:
            a = to_torch_as(a, s)
            a = a.flatten(1)
            s = torch.cat([s, a], dim=1)
        logits, h = self.preprocess(s)
        logits = self.last(logits)
        return logits


class ActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution) with MLP.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        max_action: float = 1.0,
        unbounded: bool = False,
        hidden_layer_size: int = 128,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        else:
            self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        logits, h = self.preprocess(s, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        return (mu, sigma), state


class RecurrentActorProb(nn.Module):
    """Recurrent version of ActorProb.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int],
        max_action: float = 1.0,
        unbounded: bool = False,
        hidden_layer_size: int = 128,
        conditioned_sigma: bool = False,
    ) -> None:
        super().__init__()
        self.nn = nn.LSTM(
            input_size=np.prod(state_shape),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.mu = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_layer_size, np.prod(action_shape))
        else:
            self.sigma = nn.Parameter(torch.zeros(np.prod(action_shape), 1))
        self._max = max_action
        self._unbounded = unbounded

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Dict[str, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        # s [bsz, len, dim] (training) or [bsz, dim] (evaluation)
        # In short, the tensor's shape in training phase is longer than which
        # in evaluation phase.
        if len(s.shape) == 2:
            s = s.unsqueeze(-2)
        self.nn.flatten_parameters()
        if state is None:
            s, (h, c) = self.nn(s)
        else:
            # we store the stack data in [bsz, len, ...] format
            # but pytorch rnn needs [len, bsz, ...]
            s, (h, c) = self.nn(s, (state["h"].transpose(0, 1).contiguous(),
                                    state["c"].transpose(0, 1).contiguous()))
        logits = s[:, -1]
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(
                self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX
            ).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma.view(shape) + torch.zeros_like(mu)).exp()
        # please ensure the first dim is batch size: [bsz, len, ...]
        return (mu, sigma), {"h": h.transpose(0, 1).detach(),
                             "c": c.transpose(0, 1).detach()}


class RecurrentCritic(nn.Module):
    """Recurrent version of Critic.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        layer_num: int,
        state_shape: Sequence[int],
        action_shape: Sequence[int] = [0],
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.nn = nn.LSTM(
            input_size=np.prod(state_shape),
            hidden_size=hidden_layer_size,
            num_layers=layer_num,
            batch_first=True,
        )
        self.fc2 = nn.Linear(hidden_layer_size + np.prod(action_shape), 1)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        a: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        """Almost the same as :class:`~tianshou.utils.net.common.Recurrent`."""
        assert len(s.shape) == 3
        self.nn.flatten_parameters()
        s, (h, c) = self.nn(s)
        s = s[:, -1]
        if a is not None:
            a = to_torch_as(a, s)
            s = torch.cat([s, a], dim=1)
        s = self.fc2(s)
        return s

class DistributionalCritic(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, atoms, features, layers, min_value, max_value):
        super().__init__()
        self.atoms = atoms
        self.min_value = min_value
        self.max_value = max_value

        self.net = MLP(obs_dim + action_dim, atoms, features, layers)

        self.register_buffer('z', torch.linspace(min_value, max_value, atoms))
        self.delta_z = (max_value - min_value) / (atoms - 1)

    def forward(self, obs, action, with_q=False):
        obs_action = torch.cat([obs, action], dim=-1)
        logits = self.net(obs_action)
        p = torch.softmax(logits, dim=-1)
        if with_q:
            q = torch.sum(p * self.z, dim=-1, keepdim=True)
            return p, q
        else:
            return p

    @torch.no_grad()
    def get_target(self, obs, action, reward, discount):
        p = self(obs, action) # [*B, N]

        # shift the atoms by reward
        target_z = reward + discount * self.z # [*B, N]
        target_z = torch.clamp(target_z, self.min_value, self.max_value) # [*B, N]

        # reproject the value to the nearby atoms
        target_z = target_z.unsqueeze(dim=-1) # [*B, N, 1]
        distance = torch.abs(target_z - self.z) # [*B, N, N]
        ratio = torch.clamp(1 - distance / self.delta_z, 0, 1) # [*B, N, N]
        target_p = torch.sum(p.unsqueeze(dim=-1) * ratio, dim=-2) # [*B, N]

        return target_p