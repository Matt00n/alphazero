
import copy

import jax

from envs import jumpy as jp
from envs import brax_env


def wrap_for_training(env: brax_env.Env,
                      episode_length: int = 1000,
                      action_repeat: int = 1) -> brax_env.Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
        env: environment to be wrapped
        episode_length: length of episode
        action_repeat: how many repeated actions to take per step

    Returns:
        An environment that is wrapped with Episode and AutoReset wrappers.  If the
        environment did not already have batch dimensions, it is additional Vmap
        wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    batched = False
    if hasattr(env, 'custom_tree_in_axes'):
        batch_indices, _ = jax.tree_util.tree_flatten(env.custom_tree_in_axes)
        if 0 in batch_indices:
            batched = True
    if not batched:
        env = VmapWrapper(env)
    env = AutoResetWrapper(env)
    return env


class VmapWrapper(brax_env.Wrapper):
    """Vectorizes Brax env."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        return jp.vmap(self.env.reset)(rng)

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        return jp.vmap(self.env.step)(state, action)


class EpisodeWrapper(brax_env.Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: brax_env.Env, episode_length: int,
                action_repeat: int):
        super().__init__(env)
        # For proper video speed.
        # TODO: Fix dt book keeping with action repeats so there isn't
        # async between sys and env steps.
        if hasattr(env, 'sys'):
            self.sys = copy.deepcopy(env.sys)
            self.sys.config.dt *= action_repeat
            self.sys.config.substeps *= action_repeat
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info['steps'] = jp.zeros(rng.shape[:-1])
        state.info['truncation'] = jp.zeros(rng.shape[:-1])
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jp.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jp.where(steps >= episode_length,
                                            1 - state.done, zero)
        state.info['steps'] = steps
        return state.replace(done=done)


class AutoResetWrapper(brax_env.Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: brax_env.State, action: jp.ndarray) -> brax_env.State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jp.tree_map(where_done, state.info['first_pipeline_state'], state.pipeline_state)
        obs = where_done(state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)