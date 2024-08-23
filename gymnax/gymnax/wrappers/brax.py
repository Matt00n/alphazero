from typing import Any, Dict, Union, Optional

try:
    from envs.brax_env import Env
except ImportError:
    raise ImportError("You need to install `brax` to use the brax wrapper.")
import jax
from jax import numpy as jnp
import chex
from ..environments.environment import Environment, EnvState, EnvParams
from ..environments.spaces import Discrete
from flax.struct import dataclass, field



@dataclass
class State:  # Lookalike for brax.envs.env.State
    pipeline_state: EnvState  # Brax QP is roughly equivalent to our EnvState
    obs: jax.Array # Any  # depends on environment
    reward: jax.Array # float
    done: jax.Array # bool
    real_obs: jax.Array # observation prior to resetting
    metrics: Dict[str, Union[chex.Array, chex.Scalar]] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


class GymnaxToBraxWrapper(Env):
    def __init__(self, env: Environment):
        """Wrap Gymnax environment as Brax environment

        Primarily useful for including obs, reward, and done as part of state.
        Compatible with all brax wrappers, but AutoResetWrapper is redundant since Gymnax environments
        already reset state.

        Args:
            env: Gymnax environment instance
        """
        super().__init__() # old argument: ''
        self.env = env

    def reset(self, rng: chex.PRNGKey, params: Optional[EnvParams] = None) -> State:
        """Reset, return brax State. Save rng and params in info field for step"""
        if params is None:
            params = self.env.default_params
        obs, env_state = self.env.reset(rng, params)
        return State(
            env_state,
            obs,
            0., # jnp.zeros(1)
            0., # jnp.zeros(1)
            obs,
            {},
            {"_rng": jax.random.split(rng)[0], "_env_params": params,
             'truncation': jnp.zeros(rng.shape[:-1])},
        )

    def step(
        self,
        state: State,
        action: Union[chex.Scalar, chex.Array],
        params: Optional[EnvParams] = None,
    ) -> State:
        """Step, return brax State. Update stored rng and params (if provided) in info field"""
        rng, step_rng = jax.random.split(state.info["_rng"])
        if params is None:
            params = self.env.default_params
        state.info.update(_rng=rng, _env_params=params)
        o, env_state, r, terminated, info = self.env.step(step_rng, state.pipeline_state, action, params)
        state.info['truncation'] = info['truncation']
        return state.replace(pipeline_state=env_state, obs=o, reward=r, done=terminated, real_obs=o)

    def action_size(self) -> int:
        """DEFAULT size of action vector expected by step. Can't pass params to property"""
        if isinstance(self.env.action_space(self.env.default_params), Discrete):
            return self.env.num_actions
        a_space = self.env.action_space(self.env.default_params)
        example_a = a_space.sample(jax.random.PRNGKey(0))
        return len(jax.tree_util.tree_flatten(example_a)[0])
    
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        # TODO not correct for atari etc.
        rng = jax.random.PRNGKey(0)
        reset_state = self.reset(rng)
        return reset_state.obs.shape[-1]
    
    def backend(self) -> Optional[str]:
        """The physics backend that this env was instantiated with."""
        return None
