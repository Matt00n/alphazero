# pylint:disable=g-multiple-import
"""A brax environment for training and inference."""

import abc
from typing import Any, Dict, Optional

from flax import struct
import jax

from envs.state import State, Base


@struct.dataclass
class State(Base):
    """Environment state for training and inference."""

    pipeline_state: Optional[State]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


class Env(abc.ABC):
    """Interface for driving training and inference."""

    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    @abc.abstractmethod
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""

    @property
    def unwrapped(self) -> 'Env':
        return self



class Wrapper(Env):
    """Wraps an environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, rng: jax.Array) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jax.Array) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    @property
    def backend(self) -> str:
        return self.unwrapped.backend

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)