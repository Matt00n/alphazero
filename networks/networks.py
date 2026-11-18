import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


def identity_observation_preprocessor(
    observation: jnp.ndarray, preprocessor_params: Any
):
    del preprocessor_params
    return observation


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True
    normalize_output: bool = False

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                use_bias=self.bias,
            )(hidden)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        if self.normalize_output:
            hidden = linen.LayerNorm()(hidden)
        return hidden


class AtariTorso(linen.Module):
    """ConvNet Feature Extractor."""

    layer_sizes: Sequence[int] = (512,)
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.orthogonal(jnp.sqrt(2))
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        # hidden = jnp.moveaxis(data, -3, -1) # jnp.transpose(data, (0, 2, 3, 1))
        # hidden = hidden / (255.0)

        hidden = data
        # hidden = linen.Conv(
        #     32,
        #     kernel_size=(8, 8),
        #     strides=(4, 4),
        #     padding="VALID",
        #     name='conv_1',
        #     kernel_init=self.kernel_init,
        #     bias_init=jax.nn.initializers.constant(0.0),
        # )(hidden)
        # hidden = self.activation(hidden)
        # hidden = linen.Conv(
        #     64,
        #     kernel_size=(4, 4),
        #     strides=(2, 2),
        #     padding="VALID",
        #     name='conv_2',
        #     kernel_init=self.kernel_init,
        #     bias_init=jax.nn.initializers.constant(0.0),
        # )(hidden)
        # hidden = self.activation(hidden)
        # hidden = linen.Conv(
        #     64,
        #     kernel_size=(3, 3),
        #     strides=(1, 1),
        #     padding="VALID",
        #     name='conv_3',
        #     kernel_init=self.kernel_init,
        #     bias_init=jax.nn.initializers.constant(0.0),
        # )(hidden)
        # hidden = self.activation(hidden)

        hidden = linen.Conv(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            name="conv_3",
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        hidden = self.activation(hidden)
        hidden = linen.Conv(
            64,
            kernel_size=(2, 2),
            strides=(1, 1),
            padding="VALID",
            name="conv_4",
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        hidden = self.activation(hidden)

        hidden = hidden.reshape(hidden.shape[:-3] + (-1,))
        hidden = linen.Dense(
            512,
            kernel_init=self.kernel_init,
            bias_init=jax.nn.initializers.constant(0.0),
        )(hidden)
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(
                hidden_size,
                name=f"hidden_{i}",
                kernel_init=self.kernel_init,
                bias_init=jax.nn.initializers.constant(0.0),
                use_bias=self.bias,
            )(hidden)
            hidden = self.activation(hidden)
        return hidden


def make_atari_feature_extractor(
    obs_size: int,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a CNN feature extractor."""
    feature_extractor = AtariTorso(
        layer_sizes=list(hidden_layer_sizes),
        activation=activation,
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observation_fn(obs, processor_params)
        return feature_extractor.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)
    return FeedForwardNetwork(
        init=lambda key: feature_extractor.init(key, dummy_obs), apply=apply
    )


def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observation_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)  # jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs), apply=apply
    )


def make_value_network(
    obs_size: int,
    num_atoms: int = 1,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    value_module = MLP(
        layer_sizes=list(hidden_layer_sizes) + [num_atoms],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observation_fn(obs, processor_params)
        # return jnp.squeeze(value_module.apply(policy_params, obs), axis=-1)
        return value_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)  # jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: value_module.init(key, dummy_obs), apply=apply
    )


def make_representation_function(
    obs_size: int,
    embedding_dim: int,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a MLP feature extractor."""
    representation_fn = MLP(
        layer_sizes=list(hidden_layer_sizes) + [embedding_dim],
        activation=activation,  # linen.tanh, #
        kernel_init=jax.nn.initializers.lecun_uniform(),
        normalize_output=True,
        activate_final=True,
    )

    def apply(processor_params, representation_params, obs):
        obs = preprocess_observation_fn(obs, processor_params)
        return representation_fn.apply(representation_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)
    return FeedForwardNetwork(
        init=lambda key: representation_fn.init(key, dummy_obs), apply=apply
    )


# ### HISTORY GRU (muzero Config.representation_type = "gru"): recurrent
# ### representation over the observation history of envs.pomdp_wrappers.
# ### HistoryWrapper (flat [history_length * frame_size], oldest frame first,
# ### last entry of each frame = valid flag). Frame encoder MLP -> GRU over the
# ### frames -> MLP head with LayerNorm output, like make_representation_function.
# ###   carry_size = 0: the GRU starts from a zero state at the oldest frame
# ###       (memory = the window). A pure function of the stored observation.
# ###   carry_size = R > 0 (muzero Config.rnn_carry_state): the observation is
# ###       [carry0 (R), frames]; carry0 = GRU state before the oldest frame,
# ###       computed when acting and stored with the transition (R2D2-style
# ###       stored state; the window is the burn-in, with gradients through it).
# ###       `advance` returns the next step's carry0 = one GRU step on the
# ###       oldest frame. The GRU state stays zero on padding frames (valid 0).
class HistoryGRU(linen.Module):
    embedding_dim: int
    encoder_layer_sizes: Sequence[int]
    rnn_hidden_size: int
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    def setup(self):
        self.encoder = (
            MLP(
                layer_sizes=tuple(self.encoder_layer_sizes),
                activation=self.activation,
                kernel_init=self.kernel_init,
                activate_final=True,
            )
            if self.encoder_layer_sizes
            else None
        )
        self.rnn = linen.GRUCell(features=self.rnn_hidden_size)
        self.head = MLP(
            layer_sizes=(self.embedding_dim,),
            activation=self.activation,
            kernel_init=self.kernel_init,
            normalize_output=True,
            activate_final=True,
        )

    def _step(self, carry, frame, valid):
        x = frame if self.encoder is None else self.encoder(frame)
        new_carry, _ = self.rnn(carry, x)
        return jnp.where(valid[:, None] > 0.5, new_carry, jnp.zeros_like(new_carry))

    def __call__(self, frames: jnp.ndarray, valid: jnp.ndarray, carry0: jnp.ndarray):
        # frames [N, H, F] (normalized), valid [N, H] (raw 0/1), carry0 [N, R]
        carry = carry0
        for i in range(frames.shape[1]):  # H is small: unrolled
            carry = self._step(carry, frames[:, i], valid[:, i])
        return self.head(carry)

    def advance(self, frames: jnp.ndarray, valid: jnp.ndarray, carry0: jnp.ndarray):
        return self._step(carry0, frames[:, 0], valid[:, 0])


def make_history_gru_representation_function(
    obs_size: Sequence[int],
    frame_size: int,
    embedding_dim: int,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (64, 64),
    rnn_hidden_size: int = 64,
    activation: ActivationFn = linen.relu,
    carry_state: bool = False,
) -> FeedForwardNetwork:
    """GRU representation over a flat frame history (see HistoryGRU).

    With carry_state the observation is [carry0 (rnn_hidden_size), frames] and the
    returned network has an extra `advance(processor_params, params, obs)` giving
    the next step's carry0."""
    (flat_size,) = tuple(obs_size)
    carry_size = rnn_hidden_size if carry_state else 0
    assert (flat_size - carry_size) % frame_size == 0, (flat_size, carry_size, frame_size)
    history_length = (flat_size - carry_size) // frame_size
    module = HistoryGRU(
        embedding_dim=embedding_dim,
        encoder_layer_sizes=tuple(hidden_layer_sizes),
        rnn_hidden_size=rnn_hidden_size,
        activation=activation,
    )

    def split(processor_params, obs):
        raw = jnp.reshape(obs, (-1, flat_size))
        normalized = jnp.reshape(preprocess_observation_fn(obs, processor_params), (-1, flat_size))
        frames = jnp.reshape(normalized[:, carry_size:], (-1, history_length, frame_size))
        valid = jnp.reshape(raw[:, carry_size:], (-1, history_length, frame_size))[..., -1]
        if carry_state:
            carry0 = raw[:, :carry_size]  # raw, never normalized
        else:
            carry0 = jnp.zeros((raw.shape[0], rnn_hidden_size), raw.dtype)
        return frames, valid, carry0

    def apply(processor_params, representation_params, obs):
        embedding = module.apply(representation_params, *split(processor_params, obs))
        return jnp.reshape(embedding, obs.shape[:-1] + (embedding_dim,))

    def advance(processor_params, representation_params, obs):
        carry = module.apply(
            representation_params, *split(processor_params, obs), method=HistoryGRU.advance
        )
        return jnp.reshape(carry, obs.shape[:-1] + (rnn_hidden_size,))

    dummy = (
        jnp.zeros((1, history_length, frame_size)),
        jnp.ones((1, history_length)),
        jnp.zeros((1, rnn_hidden_size)),
    )
    network = FeedForwardNetwork(init=lambda key: module.init(key, *dummy), apply=apply)
    network.advance = advance
    return network


# TODO temp oracle test, undo
def make_tmp_representation_function(
    obs_size: int,
    embedding_dim: int,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a MLP feature extractor."""
    representation_fn = MLP(
        layer_sizes=list(hidden_layer_sizes) + [embedding_dim],
        activation=activation,  # linen.tanh, #
        kernel_init=jax.nn.initializers.lecun_uniform(),
        normalize_output=True,
        activate_final=True,
    )

    def apply(processor_params, representation_params, obs):
        obs = preprocess_observation_fn(obs, processor_params)
        return obs  # representation_fn.apply(representation_params, obs)

    dummy_obs = jnp.zeros((1,) + obs_size)
    return FeedForwardNetwork(
        init=lambda key: representation_fn.init(key, dummy_obs), apply=apply
    )


class DualMLP(linen.Module):
    """MLP module."""

    layer_sizes_1: Sequence[int]
    layer_sizes_2: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        # hidden = linen.LayerNorm()(hidden)
        out1 = MLP(
            layer_sizes=self.layer_sizes_1,
            activation=self.activation,
            kernel_init=jax.nn.initializers.lecun_uniform(),
        )(hidden)
        out2 = MLP(
            layer_sizes=self.layer_sizes_2,
            activation=self.activation,  #  linen.tanh, #
            kernel_init=jax.nn.initializers.lecun_uniform(),
            normalize_output=True,
            activate_final=True,
        )(hidden)
        return out1, out2


def make_dynamics_network(
    embedding_size: int,
    num_actions: int,
    num_reward_atoms: int = 1,
    preprocess_observation_fn: Callable = identity_observation_preprocessor,
    hidden_layer_sizes_reward: Sequence[int] = (256, 256),
    hidden_layer_sizes_next_state: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    dynamics_module = DualMLP(
        layer_sizes_1=list(hidden_layer_sizes_reward) + [num_reward_atoms],
        layer_sizes_2=list(hidden_layer_sizes_next_state) + [embedding_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, dynamics_params, obs, actions):
        obs = preprocess_observation_fn(obs, processor_params)
        obs = jnp.concatenate([obs, jax.nn.one_hot(actions, num_actions)], axis=1)
        return dynamics_module.apply(dynamics_params, obs)

    dummy_obs = jnp.zeros(
        (1,) + (embedding_size + num_actions,)
    )  # jnp.zeros((1, obs_size))
    return FeedForwardNetwork(
        init=lambda key: dynamics_module.init(key, dummy_obs), apply=apply
    )
