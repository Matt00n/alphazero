"""Wrappers for partially observable environments (single, unbatched env).

Apply them directly on top of the GymnaxToBraxWrapper, i.e. inside
wrap_for_training's Episode / AutoReset wrappers. Then an auto-reset also
resets the history, `real_obs` (the observation before an auto-reset) is
transformed consistently, and the true-environment search (VmapWrapper of the
same env) steps the same observations.

ObservationSubsetWrapper  keep only some observation entries (makes a Markov
                          env partially observable, e.g. Acrobot without the
                          two angular velocities: indices (0, 1, 2, 3)).
HistoryWrapper            observation = the last `history_length` frames,
                          oldest first, flattened. Each frame is
                          [obs, one-hot of the action that led to it, valid],
                          valid = 0 for padding before the episode start.
                          MuZero's representation input (32 frames + actions
                          on Atari). Stateless: the history lives in the
                          observation itself, so the replay buffer, reanalyze
                          and the losses need no changes.
RecurrentStateSlotWrapper info["rnn_carry"] slot for a carried recurrent
                          agent state (see the class).
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from envs.brax_env import Wrapper


class ObservationSubsetWrapper(Wrapper):
    def __init__(self, env, indices: Sequence[int]):
        super().__init__(env)
        self.indices = jnp.asarray(tuple(indices), dtype=jnp.int32)

    def _subset(self, state):
        return state.replace(obs=state.obs[..., self.indices], real_obs=state.real_obs[..., self.indices])

    def reset(self, rng):
        return self._subset(self.env.reset(rng))

    def step(self, state, action):
        # the inner (gymnax) env does not read state.obs, only pipeline_state / info
        return self._subset(self.env.step(state, action))

    def observation_size(self) -> int:
        return int(self.indices.shape[0])


class HistoryWrapper(Wrapper):
    def __init__(self, env, history_length: int, num_actions: int, include_actions: bool = True):
        super().__init__(env)
        self.history_length = history_length
        self.num_actions = num_actions
        self.include_actions = include_actions

    def frame_size(self, obs_size: int) -> int:
        return obs_size + (self.num_actions if self.include_actions else 0) + 1

    def _frame(self, obs, action, valid):
        parts = [obs]
        if self.include_actions:
            parts.append(action)
        parts.append(jnp.reshape(valid, (1,)).astype(obs.dtype))
        return jnp.concatenate(parts, axis=-1)

    def reset(self, rng):
        state = self.env.reset(rng)
        frame = self._frame(state.obs, jnp.zeros(self.num_actions, state.obs.dtype), 1.0)
        history = jnp.zeros((self.history_length, frame.shape[-1]), frame.dtype).at[-1].set(frame)
        history = jnp.reshape(history, (-1,))
        return state.replace(obs=history, real_obs=history)

    def step(self, state, action):
        previous = state.obs
        nstate = self.env.step(state, action)
        frame = self._frame(nstate.obs, jax.nn.one_hot(action, self.num_actions, dtype=nstate.obs.dtype), 1.0)
        history = jnp.concatenate([previous[frame.shape[-1]:], frame], axis=-1)
        return nstate.replace(obs=history, real_obs=history)

    def observation_size(self) -> int:
        return self.history_length * self.frame_size(self.env.observation_size())


class RecurrentStateSlotWrapper(Wrapper):
    """Adds info["rnn_carry"] (zeros of `size`) at reset and keeps it on step.

    Used with a carried recurrent representation (muzero Config.rnn_carry_state):
    the agent (actor_step) writes its recurrent state there after every step, so
    every loop that carries env states (acting, prefill, evaluation) carries the
    agent's recurrent state too. Auto-resets keep the info of the stepped state;
    actor_step zeroes the state where an episode ended."""

    def __init__(self, env, size: int):
        super().__init__(env)
        self.size = size

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["rnn_carry"] = jnp.zeros((self.size,), state.obs.dtype)
        return state
