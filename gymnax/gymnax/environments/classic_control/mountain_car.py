import jax
import jax.numpy as jnp
from jax import lax
from gymnax.gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    position: float
    velocity: float
    time: int


@struct.dataclass
class EnvParams:
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.5
    goal_velocity: float = 0.0
    force: float = 0.001
    gravity: float = 0.0025
    max_steps_in_episode: int = 5000


class MountainCar(environment.Environment):
    """
    JAX Compatible  version of MountainCar-v0 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        velocity = (
            state.velocity
            + (action - 1) * params.force
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (
            1 - (position == params.min_position) * (velocity < 0)
        )

        reward = -1.0

        # Update state dict and evaluate termination conditions
        state = EnvState(position, velocity, state.time + 1)
        done, truncated = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params),
             "truncation": truncated},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_state, velocity=0.0, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        return jnp.array([state.position, state.velocity])

    def is_terminal(self, state: EnvState, params: EnvParams) -> Tuple[bool, bool]:
        """Check whether state is terminal."""
        done = jnp.array((state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )).astype(float)

        # Check number of steps in episode termination condition
        truncated = jnp.array(state.time >= params.max_steps_in_episode).astype(float)
        return done, truncated

    @property
    def name(self) -> str:
        """Environment name."""
        return "MountainCar-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, (2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )

        return spaces.Dict(
            {
                "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
                "velocity": spaces.Box(low[1], high[1], (), dtype=jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
