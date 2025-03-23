"""AlphaZero reimplementation."""

from absl import logging, app
from functools import partial
import os
import pickle
import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

# os.environ[
#     "XLA_PYTHON_CLIENT_MEM_FRACTION"
# ] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as tfd # TODO: can be removed once transitioned to AZ loss

from envs import make_env, Transition, MCTSTransition, has_discrete_action_space, is_atari_env
# from envs.brax_v1_wrappers import wrap_for_training
from envs.brax_wrappers import EvalWrapper, wrap_for_training, VmapWrapper
from networks.policy import Policy, ForwardPass
from networks.networks import FeedForwardNetwork, ActivationFn, make_policy_network, make_value_network, make_atari_feature_extractor
from networks.distributions import NormalTanhDistribution, ParametricDistribution, PolicyNormalDistribution, DiscreteDistribution
import replay_buffers
import running_statistics
from gymnax import gymnax
from gymnax.gymnax.wrappers.brax import GymnaxToBraxWrapper, State
import mctx_dist as mctx


class Config:
    # experiment
    experiment_name = 'base_short'
    seed = 20
    platform = 'cpu' # CPU or GPU
    capture_video = False
    write_logs_to_file = False
    save_model = False
    save_scores = False

    # environment
    env_id = 'Acrobot-v1' # CartPole-v1, Breakout-MinAtar, MountainCar-v0, Acrobot-v1
    num_envs = 16
    normalize_observations = True 
    action_repeat = 1
    eval_env = True
    num_resets_per_eval = 0
    eval_every = 5
    deterministic_eval = True
    num_eval_envs = 64 
    episode_length = 500

    # MCTS
    num_simulations = 30
    max_num_considered_actions = 16
    l2_coef = 1e-4 # 1e-4
    vf_cost = 0.5
    use_gae = True
    gae_lambda = 0.95
    n_step_gamma = 0.99 # 0.99
    n_step_n = 5

    # quantile regression
    num_atoms: int = 1 # 8 # NOTE: num_atoms = 1 yields non-distributional RL with MSE loss
    qr_kappa: float = 1.

    # reanalyze
    reanalyze: bool = False

    # replay buffer
    min_replay_size: int = 8192
    max_replay_size: Optional[int] = 8192 # 8192 # 16384
    replay_buffer_batch_size: int = 128 # 256
    per_alpha: float = 0.
    per_importance_sampling: bool = True
    per_importance_sampling_beta: float = 1.

    # algorithm hyperparameters
    total_timesteps = int(1e6) 
    learning_rate = 1e-3 # 3e-4 
    unroll_length = 128 # 512 # 128 
    anneal_lr = True
    num_minibatches = 128 # 64
    update_epochs = 1 # NOTE: no reason to increase this, just increase sample size instead
    max_grad_norm = 0.5
    
    # policy params
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4 
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5 
    activation: ActivationFn = nn.swish 
    squash_distribution: bool = True

    # atari params
    atari_dense_layer_sizes: Sequence[int] = (256,) # (512,)

InferenceParams = Tuple[running_statistics.NestedMeanStd, Any] # Not used
Metrics = Mapping[str, jnp.ndarray]
ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # in order to avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


@flax.struct.dataclass
class AZNetworkParams:
    """Contains training state for the learner."""
    policy: Any
    value: Any


@flax.struct.dataclass
class AZNetworks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class AtariAZNetworkParams:
    """Contains training state for the learner."""
    feature_extractor: Any
    policy: Any
    value: Any


@flax.struct.dataclass
class AtariAZNetworks:
    feature_extractor: FeedForwardNetwork
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Union[AZNetworkParams, AtariAZNetworkParams]
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def make_inference_fn(az_networks: Union[AZNetworks, AtariAZNetworks]):
    """Creates params and inference function for the agent."""

    def make_policy(params: Any,
                    deterministic: bool = False,
                    use_feature_extractor: bool = False,
                    ) -> Policy:
        policy_network = az_networks.policy_network
        parametric_action_distribution = az_networks.parametric_action_distribution
        if use_feature_extractor:
            shared_feature_extractor = az_networks.feature_extractor
        normalizer_params, policy_params, feature_extractor_params = params

        @jax.jit # TODO jit needed ?
        def policy(observations: jnp.ndarray,
                key_sample: jnp.ndarray) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            if use_feature_extractor:
                observations = shared_feature_extractor.apply(normalizer_params, feature_extractor_params, observations)
            logits = policy_network.apply(normalizer_params, policy_params, observations)
            if deterministic:
                return az_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions)
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions
            }

        return policy

    return make_policy


def make_forward_fn(az_networks: Union[AZNetworks, AtariAZNetworks]):
    """Creates params and inference function for the agent."""

    def make_forward(params: Any,
                    use_feature_extractor: bool = False,
                    ) -> ForwardPass:
        policy_network = az_networks.policy_network
        value_network = az_networks.value_network
        if use_feature_extractor:
            shared_feature_extractor = az_networks.feature_extractor
        normalizer_params, policy_params, value_params, feature_extractor_params = params

        @jax.jit # TODO jit needed ?
        def forward_networks(observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            if use_feature_extractor:
                observations = shared_feature_extractor.apply(normalizer_params, feature_extractor_params, observations)
            
            logits = policy_network.apply(normalizer_params, policy_params, observations)
            value = value_network.apply(normalizer_params, value_params, observations)

            return logits, value

        return forward_networks

    return make_forward


def make_az_networks(
        observation_size: Union[Sequence[int], int],
        action_size: int,
        num_atoms: int,
        preprocess_observation_fn: Callable, 
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: ActivationFn = nn.swish,
        sqash_distribution: bool = True,
        discrete_policy: bool = False,
        shared_feature_extractor: bool = False,
        feature_extractor_dense_hidden_layer_sizes: Optional[Sequence[int]] = (512,),
    ) -> Union[AZNetworks, AtariAZNetworkParams]:
    """Make networks with preprocessor."""
    if discrete_policy:
        parametric_action_distribution = DiscreteDistribution(
            param_size=action_size)
    elif sqash_distribution:
        parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size)
    else:
        parametric_action_distribution = PolicyNormalDistribution(
            event_size=action_size)
    if shared_feature_extractor:
        feature_extractor = make_atari_feature_extractor(
            obs_size=observation_size,
            preprocess_observation_fn=preprocess_observation_fn,
            hidden_layer_sizes=feature_extractor_dense_hidden_layer_sizes,
            activation=nn.relu
        )
        policy_network = make_policy_network(
            parametric_action_distribution.param_size,
            feature_extractor_dense_hidden_layer_sizes[-1:],
            hidden_layer_sizes=(),
            activation=activation)
        value_network = make_value_network(
            feature_extractor_dense_hidden_layer_sizes[-1:],
            num_atoms=num_atoms,
            hidden_layer_sizes=(),
            activation=activation)
        return AtariAZNetworks(
            feature_extractor=feature_extractor,
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=parametric_action_distribution)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observation_fn=preprocess_observation_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation)
    value_network = make_value_network(
        observation_size,
        num_atoms=num_atoms,
        preprocess_observation_fn=preprocess_observation_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation)

    return AZNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


def actor_step(
    env: GymnaxToBraxWrapper,
    rollout_env: Any,
    env_state: State,
    forward: ForwardPass,
    key: jnp.ndarray,
    deterministic_actions: bool = False,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, MCTSTransition]:
    """Collect data."""

    key, logits_rng, search_rng = jax.random.split(key, 3)

    # logits at root produced by the prior policy 
    prior_logits, value = forward(env_state.obs)
   
    use_mixed_value = False

    # NOTE: For AlphaZero embedding is env_state, for MuZero
    # the root output would be the output of MuZero representation network.
    root = mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        # The embedding is used only to implement the MuZero model.
        embedding=env_state, 
    )

    # The recurrent_fn is provided by MuZero dynamics network.
    # Or true environment for AlphaZero
    # TODO MCTS: pass in dynamics function for MuZero
    def recurrent_fn(params, rng_key, action, embedding):
        # environment (model)
        env_state = embedding
        nstate = rollout_env.step(env_state, action)

        # policy & value networks
        prior_logits, value = forward(nstate.obs) # priorly: env_state

        # Create the new MCTS node.
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=nstate.reward,
            # discount when terminal state reached
            discount= Config.n_step_gamma * jnp.where(
                nstate.info['truncation'], jnp.ones_like(nstate.done), 1 - nstate.done
            ), # 1 - nstate.done, 
            # prior for the new state
            prior_logits=prior_logits,
            # value for the new state
            value=value,
        )

        # Return the new node and the new environment.
        return recurrent_fn_output, nstate

    # Running the search.
    # policy_output = mctx.gumbel_muzero_policy(
    #     params=(),
    #     rng_key=search_rng,
    #     root=root,
    #     recurrent_fn=recurrent_fn,
    #     num_simulations=Config.num_simulations,
    #     max_num_considered_actions=Config.max_num_considered_actions,
    #     qtransform=partial(
    #         mctx.qtransform_completed_by_mix_value,
    #         use_mixed_value=use_mixed_value),
    # )

    policy_output = mctx.sampled_muzero_policy( # muzero_policy
        params=(),
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=Config.num_simulations,
        dirichlet_fraction=0.25,
        dirichlet_alpha=0.3,
        pb_c_init=1.25,# 1.25,
        pb_c_base=19652, # 19652,
        temperature=1.0,
    )

    actions = policy_output.action
    action_weights = policy_output.action_weights
    # best_actions = jnp.argmax(action_weights, axis=-1).astype(jnp.int32)
    qvalues = jnp.mean(policy_output.search_tree.summary().qvalues, axis=-1)
    masked_qvalues = jnp.where(action_weights, qvalues, -jnp.inf)
    best_actions = jnp.argmax(masked_qvalues, axis=-1).astype(jnp.int32)
    actions = jax.lax.select(deterministic_actions, best_actions, actions)
    
    search_value = policy_output.search_tree.summary().value
    # search_value = policy_output.search_tree.summary().qvalues[jnp.arange(actions.shape[0]), actions, :]

    policy_extras = {
        'prior_log_prob': tfd.Categorical(logits=prior_logits).log_prob(actions),
        'raw_action': actions
    }

    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, MCTSTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        real_obs=env_state.real_obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.real_obs, # nstate.obs
        target_policy_probs=action_weights,
        search_value=search_value,
        value_prefix_target=jnp.zeros_like(nstate.reward),
        bootstrap_observation=jnp.zeros_like(env_state.obs),
        bootstrap_value=value, # NOTE: we overload here to save space
        bootstrap_discount=jnp.ones_like(1 - nstate.done),
        extras={
            'policy_extras': policy_extras, 
            'state_extras': state_extras
        },
        priority=jnp.ones_like(nstate.done),
        weight=jnp.ones_like(nstate.done),) 


def generate_unroll(
    env: GymnaxToBraxWrapper,
    rollout_env: Any,
    env_state: State,
    forward: ForwardPass,
    key: jnp.ndarray,
    unroll_length: int,
    deterministic_actions: bool = False,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, MCTSTransition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(
            env, rollout_env, state, forward, current_key, 
            deterministic_actions=deterministic_actions, 
            extra_fields=extra_fields)
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length)
    return final_state, data


class Evaluator:
    """Class to run evaluations."""

    def __init__(self, eval_env: GymnaxToBraxWrapper,
                rollout_env: Any,
                eval_forward_fn: Callable[[Any],
                                            Policy], num_eval_envs: int,
                episode_length: int, action_repeat: int, key: jnp.ndarray,
                deterministic_eval: bool = True):
        """Init.

        Args:
            eval_env: Batched environment to run evals on.
            eval_policy_fn: Function returning the policy from the policy parameters.
            num_eval_envs: Each env will run 1 episode in parallel for each eval.
            episode_length: Maximum length of an episode.
            action_repeat: Number of physics steps per env step.
            key: RNG key.
            deterministic_eval: whether to choose actions deterministically.
        """
        self._key = key
        self._eval_walltime = 0.

        eval_env = EvalWrapper(eval_env)

        def generate_eval_unroll(policy_params: Any,
                                key: jnp.ndarray) -> State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                rollout_env,
                eval_first_state,
                eval_forward_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
                deterministic_actions=deterministic_eval,
                )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self,
                        policy_params: Any,
                        training_metrics: Metrics,
                        aggregate_episodes: bool = True) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std]:
            suffix = '_std' if fn == np.std else ''
            metrics.update(
                {
                    f'eval/episode_{name}{suffix}': (
                        float(fn(value)) if aggregate_episodes else value
                    )
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )
        metrics['eval/avg_episode_length'] = float(np.mean(eval_metrics.episode_steps))
        metrics['eval/epoch_eval_time'] = np.round(epoch_eval_time)
        metrics['eval/sps'] = np.round(self._steps_per_unroll / epoch_eval_time)
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            'eval/walltime': self._eval_walltime,
            **training_metrics,
            **metrics
        }

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
    

def reanalyze(data: MCTSTransition, 
              forward: ForwardPass,
              env: GymnaxToBraxWrapper,
              rollout_env: Any,
              key: jnp.ndarray) -> MCTSTransition:
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (Config.num_minibatches, -1) 
                                                        + x.shape[1:]), data)
    
    def f(carry_key, data: MCTSTransition):
        new_key, key = jax.random.split(carry_key)
        # TODO: env_state currently not saved! needs to be saved or restored from other data!
        env_state = data.env_state
        _, reanalyzed_data = actor_step(env, rollout_env, env_state, forward, key)

        return new_key, reanalyzed_data
        
    _, reanalyzed_data = jax.lax.scan(
        f, 
        key,
        data,
        length=Config.num_minibatches)
    
    chex.assert_equal_shape(data.target_policy_probs, reanalyzed_data.target_policy_probs)
    chex.assert_equal_shape(data.bootstrap_value, reanalyzed_data.bootstrap_value)
    
    data = data._replace(target_policy_probs=reanalyzed_data.target_policy_probs,
                         bootstrap_value=reanalyzed_data.bootstrap_value)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
    
    return data
    

def n_step_bootstrapped_targets(
        rewards: jnp.ndarray,
        discounts: jnp.ndarray,
        termination_discount: jnp.ndarray,
        observations: jnp.ndarray,
        values: jnp.ndarray,
        n: int = 5,
        gamma: float = 1.,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes n-step bootstrapped return targets over a sequence.

    Args:
        rewards: rewards at times [1, ..., T].
        discounts: discounts at times [1, ..., T].
        termination_discount: discount from termination at times [1, ..., T].
        observations: observation at time [1, ...., T].
        values: values at time [1, ...., T, T+1]. We don't need the first value.
        n: number of steps over which to accumulate reward before bootstrapping.

    Returns:
        estimated bootstrapped returns prefixes at times [0, ...., T-1]
        observation to bootstrap from at times [0, ...., T-1]
        values of bootstrap observations
        discount factor for bootstrap value at times [0, ...., T-1]
    """
    values = values[1:]
    chex.assert_type([rewards, discounts, values], float)
    chex.assert_equal_shape_prefix([rewards, discounts, values], 1)
    batch_shape = rewards.shape
    seq_len = batch_shape[0]

    # Shift bootstrap values by n and pad end of sequence with last value v_t[-1].
    pad_size = min(n - 1, seq_len)
    bootstrap_observations = jnp.concatenate([observations[n - 1:], jnp.array([observations[-1]] * pad_size)])
    bootstrap_values = jnp.concatenate([values[n - 1:], jnp.array([values[-1]] * pad_size)])

    # Pad sequences. Shape is now (T + n - 1, ...).
    rewards = jnp.concatenate([rewards, jnp.zeros((n - 1,) + batch_shape[1:])])
    discounts = jnp.concatenate([discounts, jnp.ones((n - 1,) + batch_shape[1:])]) * gamma

    value_prefix_targets = jax.lax.dynamic_slice_in_dim(rewards, n-1, seq_len)
    bootstrap_discounts = jnp.concatenate([termination_discount, jnp.ones((n - 1,) + batch_shape[1:])]) * gamma
    bootstrap_discounts = jax.lax.dynamic_slice_in_dim(bootstrap_discounts, n-1, seq_len)

    def f(carry, unused_t):
        i, value_prefix_targets, bootstrap_discounts = carry
        i -= 1
        r_ = jax.lax.dynamic_slice_in_dim(rewards, i, seq_len)
        discount_ = jax.lax.dynamic_slice_in_dim(discounts, i, seq_len)
        value_prefix_targets = r_ + discount_ * value_prefix_targets
        bootstrap_discounts *= discount_
        return (i, value_prefix_targets, bootstrap_discounts), unused_t

    (_, value_prefix_targets, bootstrap_discounts), _ = jax.lax.scan(
        f, (n-1, value_prefix_targets, bootstrap_discounts),
        (),
        length=n-1)

    return value_prefix_targets, bootstrap_observations, bootstrap_values, bootstrap_discounts


def compute_gae(
        rewards: jnp.ndarray,
        discounts: jnp.ndarray,
        termination_discount: jnp.ndarray,
        observations: jnp.ndarray,
        values: jnp.ndarray,
        lambda_: float = 1.0,
        discount: float = 0.99,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
            following the behaviour policy.
        discounts: discounts at times [1, ..., T].
        termination_discount: discount from termination at times [1, ..., T].
        observations: observation at time [1, ...., T].
        values: values at time [1, ...., T, T+1]. Final value is only used as bootstrap target
        lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
            lambda_=1.
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """

    termination = jnp.expand_dims(1 - termination_discount, -1)
    truncation = jnp.expand_dims((1 - discounts), -1) * (1 - termination)

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = values[1:]
    deltas = jnp.expand_dims(rewards, -1) + discount * (1 - termination) * values_t_plus_1 - values[:-1]
    deltas *= truncation_mask

    acc = jnp.zeros_like(values[0])
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values[:-1])

    # vs_t_plus_1 = jnp.concatenate(
    #     [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    # advantages = (rewards + discount *
    #                 (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jnp.zeros_like(rewards), jnp.zeros_like(observations), vs, jnp.ones_like(discounts)



def quantile_regression_loss(
        values: jnp.ndarray,
        targets: jnp.ndarray,
        kappa: float,
        num_atoms: int,
    ) -> jnp.ndarray:
    """Quantile regression loss from "Distributional Reinforcement Learning with Quantile
    Regression" - Dabney et. al, 2017".

    Args:
        values: Predicted values for each quantile.
        targets: Target values for each quantile.
        kappa: Huber loss cutoff.
        num_atoms: Number of buckets for the value function distribution.

    Returns:
        The computed value loss.
    """
    # Input `u' of Eq. 9.
    bellman_errors = jnp.expand_dims(targets, -2) - jnp.expand_dims(values, -1)

    # Eq. 9 of paper.
    huber_loss = (
        (jnp.abs(bellman_errors) <= kappa).astype(jnp.float32) *
        0.5 * bellman_errors ** 2 +
        (jnp.abs(bellman_errors) > kappa).astype(jnp.float32) *
        kappa * (jnp.abs(bellman_errors) - 0.5 * kappa))

    tau_hat = ((jnp.arange(num_atoms, dtype=jnp.float32) + 0.5) /
               num_atoms)  # Quantile midpoints.  See Lemma 2 of paper.
    # expand tau_hat to shape of bellman errors
    tau_hat = jnp.expand_dims(tau_hat, (0, -1))

    # Eq. 10 of paper.
    tau_bellman_diff = jnp.abs(
        tau_hat - (bellman_errors < 0).astype(jnp.float32))
    quantile_huber_loss = tau_bellman_diff * huber_loss
    # Sum over tau dimension, average over target value dimension.
    loss = jnp.sum(jnp.mean(quantile_huber_loss, 2), 1)
    return loss


def mse_value_loss(values: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """MSE loss for value function.

    Args:
        values: Predicted values for each quantile.
        targets: Target values for each quantile.

    Returns:
        The computed value loss.
    """
    v_error = jax.lax.stop_gradient(targets) - values
    return jnp.squeeze(v_error * v_error * 0.5)


def compute_muzero_loss(
    params: Union[AZNetworkParams, AtariAZNetworkParams],
    normalizer_params: Any,
    data: MCTSTransition,
    rng: jnp.ndarray,
    az_network: Union[AZNetworks, AtariAZNetworks],
    value_loss_fn: Callable[[Any], jnp.ndarray],
    vf_cost: float = 0.5,
    l2_coef: float = 1e-4,
    shared_feature_extractor: bool = False,
    per_importance_sampling: bool = True,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes Alphazero loss .

    Args:
        params: Network parameters.
        normalizer_params: Parameters of the normalizer.
        data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
            ['policy_extras']['log_prob']
        rng: Random key
        az_network: networks.
        vf_cost: Value loss coefficient.
        l2_coef: L2 penalty coefficient.
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """
    parametric_action_distribution = az_network.parametric_action_distribution
    
    policy_apply = az_network.policy_network.apply
    value_apply = az_network.value_network.apply

    # Put the time dimension first.
    # data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    if shared_feature_extractor:
        feature_extractor_apply = az_network.feature_extractor.apply
        hidden = feature_extractor_apply(normalizer_params, params.feature_extractor, data.observation)
    
    policy_logits = policy_apply(normalizer_params, params.policy,
                                hidden)

    baseline = value_apply(normalizer_params, params.value, hidden)

    policy_targets = data.target_policy_probs

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action'])
    behaviour_action_log_probs = data.extras['policy_extras']['prior_log_prob']

    log_ratio = target_action_log_probs - behaviour_action_log_probs
    rho_s = jnp.exp(log_ratio)
    approx_kl = ((rho_s - 1) - log_ratio).mean()

    policy_loss = -jnp.mean(jnp.sum(jax.lax.stop_gradient(policy_targets)*(jax.nn.log_softmax(policy_logits)), axis=-1))

    # Value function loss
    vs = jnp.expand_dims(data.value_prefix_target, -1) + jnp.expand_dims(data.bootstrap_discount, -1) * data.bootstrap_value
    v_losses = value_loss_fn(baseline, jax.lax.stop_gradient(vs))
    if per_importance_sampling:
        v_losses *= data.weight
    v_loss = vf_cost * jnp.mean(v_losses)

    # l2 penalty
    l2_penalty = l2_coef * 0.5 * sum(
        jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(params))

    # Entropy
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))

    total_loss = policy_loss + v_loss + l2_penalty

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': v_loss,
        'l2_penalty': l2_penalty,
        'entropy': entropy, 
        'approx_kl': jax.lax.stop_gradient(approx_kl), 
    }

    return total_loss, metrics



def main(_):
    start_time = time.time()

    run_name = f"Exp_{Config.experiment_name}__{Config.env_id}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        from absl import flags
        flags.FLAGS.alsologtostderr = True
        log_path = f'./training_logs/az/{run_name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logging.get_absl_handler().use_absl_log_file('logs', log_path)

    logging.get_absl_handler().setFormatter(None)

    # jax set up devices
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    assert Config.num_envs % device_count == 0

    if Config.min_replay_size >= Config.total_timesteps:
        raise ValueError(
            'No training will happen because min_replay_size >= total_timesteps')

    if Config.max_replay_size is None:
        max_replay_size = Config.total_timesteps
    else:
        max_replay_size = Config.max_replay_size

    
    env_steps_per_actor_step = Config.action_repeat * Config.num_envs
    num_prefill_actor_steps = np.ceil(Config.min_replay_size / (env_steps_per_actor_step))
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert Config.total_timesteps - num_prefill_env_steps >= 0

    # The number of environment steps executed for every training step.
    env_step_per_training_step = Config.unroll_length * Config.num_envs
    num_training_steps = np.ceil(Config.total_timesteps / env_step_per_training_step).astype(int)
    num_evals_after_init = max(np.floor(num_training_steps / Config.eval_every).astype(int), 1)
    num_training_steps_per_epoch = np.ceil(
        (Config.total_timesteps - num_prefill_env_steps) / 
        (num_evals_after_init * env_step_per_training_step * max(Config.num_resets_per_eval, 1))
    ).astype(int)

    # log hyperparameters
    logging.info("|param: value|")
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            logging.info(f"|{key}:  {value}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    # handle / split random keys
    key = jax.random.PRNGKey(Config.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, rb_key, key_envs, eval_key = jax.random.split(local_key, 4)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, key_feature_extractor = jax.random.split(global_key, 3)
    del global_key
 
    # create env
    is_atari = is_atari_env(Config.env_id)
    environment, env_params = gymnax.make(Config.env_id)
    discrete_action_space = has_discrete_action_space(environment, env_params)
    if not discrete_action_space:
        raise NotImplementedError('Currently only discrete action spaces are supported.')
    environment = GymnaxToBraxWrapper(environment)

    env = wrap_for_training(
        environment,
        episode_length=Config.episode_length,
        action_repeat=Config.action_repeat,
    )
    model_rollout_env = VmapWrapper(environment)

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_envs, Config.num_envs // process_count)
    key_envs = jnp.reshape(key_envs,
                            (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    action_size = env.action_size()

    if is_atari:
        observation_shape = env_state.obs.shape[-3:]
    else:
        observation_shape = env_state.obs.shape[-1:]

    # intialize replay buffer
    dummy_obs = jnp.zeros(observation_shape,)
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = MCTSTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        real_obs=dummy_obs,
        action=0.,
        reward=0.,
        discount=0.,
        next_observation=dummy_obs,
        target_policy_probs=jnp.zeros((action_size,)),
        search_value=jnp.zeros(Config.num_atoms),
        value_prefix_target=0.,
        bootstrap_observation=dummy_obs,
        bootstrap_value=jnp.zeros(Config.num_atoms),
        bootstrap_discount=0.,
        extras={
            'state_extras': {
                'truncation': 0.
            },
            'policy_extras': {
                'prior_log_prob': 0.,
                'raw_action': 0.
            }
        },
        priority=0.,
        weight=0.,)
    
    if Config.per_alpha > -1:
        replay_buffer = replay_buffers.PrioritizedSamplingQueue( # UniformSamplingQueue Queue PrioritizedSamplingQueue
            max_replay_size=max_replay_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=Config.replay_buffer_batch_size * Config.num_minibatches // device_count,
            per_importance_sampling_beta=Config.per_importance_sampling_beta)
    else: 
        replay_buffer = replay_buffers.UniformSamplingQueue( # UniformSamplingQueue Queue PrioritizedSamplingQueue
            max_replay_size=max_replay_size // device_count,
            dummy_data_sample=dummy_transition,
            sample_batch_size=Config.replay_buffer_batch_size * Config.num_minibatches // device_count)
    

    # Normalize moved from env to network
    normalize = lambda x, y: x
    if Config.normalize_observations:
        normalize = running_statistics.normalize

    az_network = make_az_networks(
        observation_size=observation_shape, # NOTE only works with flattened observation space
        action_size=action_size, # flatten action size for nested spaces
        num_atoms=Config.num_atoms,
        preprocess_observation_fn=normalize, 
        policy_hidden_layer_sizes=Config.policy_hidden_layer_sizes, 
        value_hidden_layer_sizes=Config.value_hidden_layer_sizes,
        activation=Config.activation,
        sqash_distribution=Config.squash_distribution,
        discrete_policy=discrete_action_space,
        shared_feature_extractor=is_atari,
        feature_extractor_dense_hidden_layer_sizes=Config.atari_dense_layer_sizes,
    )
    make_forward = make_forward_fn(az_network)
    make_forward = partial(make_forward, use_feature_extractor=is_atari)

    # create optimizer
    if Config.anneal_lr:    
        learning_rate = optax.linear_schedule(
            Config.learning_rate, 
            Config.learning_rate * 0.01, # 0
            transition_steps=Config.total_timesteps, 
        )
    else:
        learning_rate = Config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.max_grad_norm),
        optax.adam(learning_rate),
    )

    if Config.use_gae:
        n_step_fn = partial(compute_gae, lambda_=Config.gae_lambda, discount=Config.n_step_gamma)
    else:
        n_step_fn = partial(n_step_bootstrapped_targets, n=Config.n_step_n, gamma=Config.n_step_gamma)
    

    if Config.num_atoms > 1:
        value_loss_fn = partial(quantile_regression_loss, kappa=Config.qr_kappa, num_atoms=Config.num_atoms)
    elif Config.num_atoms == 1:
        value_loss_fn = mse_value_loss
    else:
        raise ValueError('num_atoms must be a positive integer.')

    if Config.reanalyze:
        raise NotImplementedError
        if Config.use_gae:
            raise ValueError('Reanalyze not compatible with Generalized Advantage Estimation')
        # reanalyze_fn = reanalyze
    else:
        reanalyze_fn = lambda x, a, b, c: x

    # create loss function via functools.partial
    loss_fn = partial(
        compute_muzero_loss,
        az_network=az_network,
        value_loss_fn=value_loss_fn,
        vf_cost=Config.vf_cost,
        l2_coef=Config.l2_coef,
        shared_feature_extractor=is_atari,
        per_importance_sampling=Config.per_importance_sampling,
    )


    def loss_and_pgrad(loss_fn: Callable[..., float],
                        pmap_axis_name: Optional[str],
                        has_aux: bool = False):
        g = jax.value_and_grad(loss_fn, has_aux=has_aux)

        def h(*args, **kwargs):
            value, grad = g(*args, **kwargs)
            return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

        return g if pmap_axis_name is None else h
    

    def gradient_update_fn(loss_fn: Callable[..., float],
                            optimizer: optax.GradientTransformation,
                            pmap_axis_name: Optional[str],
                            has_aux: bool = False):
        """Wrapper of the loss function that apply gradient updates.

        Args:
            loss_fn: The loss function.
            optimizer: The optimizer to apply gradients.
            pmap_axis_name: If relevant, the name of the pmap axis to synchronize
            gradients.
            has_aux: Whether the loss_fn has auxiliary data.

        Returns:
            A function that takes the same argument as the loss function plus the
            optimizer state. The output of this function is the loss, the new parameter,
            and the new optimizer state.
        """
        loss_and_pgrad_fn = loss_and_pgrad(
            loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

        def f(*args, optimizer_state):
            value, grads = loss_and_pgrad_fn(*args)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], params_update)
            return value, params, optimizer_state

        return f
    
    gradient_update_fn = gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    # minibatch training step
    def minibatch_step(carry, data: MCTSTransition, normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics


    # sgd step
    def sgd_step(carry, unused_t, data: MCTSTransition, normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x) # TODO unnecessary: data already randomly sampled from buffer
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(minibatch_step, normalizer_params=normalizer_params), 
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (optimizer_state, params, key), metrics
    

    def training_step(
        carry: Tuple[TrainingState, State, ReplayBufferState, jnp.ndarray],
        unused_t
    ) -> Tuple[Tuple[TrainingState, State, ReplayBufferState, jnp.ndarray], Metrics]:
        training_state, state, buffer_state, key = carry
        key_sgd, key_generate_unroll, key_bootstrap, reanalyze_key, new_key = jax.random.split(key, 5)

        forward_fn = make_forward(
            (training_state.normalizer_params, training_state.params.policy, 
             training_state.params.value, training_state.params.feature_extractor)
        )

        state, data = generate_unroll(
            env,
            model_rollout_env,
            state,
            forward_fn,
            key_generate_unroll,
            Config.unroll_length,
            deterministic_actions=False,
            extra_fields=('truncation',))
        
        # additional search at final step for bootstrap values
        _, transition = actor_step(
            env, model_rollout_env, state, forward_fn, key_bootstrap, 
            deterministic_actions=False, 
            extra_fields=('truncation',))
        
        assert data.discount.shape[0] == Config.unroll_length

        value_prefix_targets, bootstrap_observations, bootstrap_values, bootstrap_discounts = n_step_fn(
            rewards=data.reward,
            discounts=data.discount * (1 - data.extras['state_extras']['truncation']),
            termination_discount=data.discount,
            observations=data.next_observation,
            values=jnp.concatenate([data.search_value, jnp.array([transition.search_value])]),
        )
        # value_prefix_targets = jnp.zeros_like(value_prefix_targets)
        # bootstrap_discounts = jnp.ones_like(bootstrap_discounts)
        # bootstrap_values = data.search_value

        # NOTE: data.bootstap_value is overloaded with the prior values
        targets = jnp.expand_dims(value_prefix_targets, -1) + jnp.expand_dims(bootstrap_discounts, -1) * bootstrap_values
        priorities = (jnp.mean(jnp.abs(targets - data.bootstrap_value), axis=-1) + 1e-10)**Config.per_alpha

        data = data._replace(value_prefix_target=value_prefix_targets, bootstrap_observation=bootstrap_observations,
                      bootstrap_value=bootstrap_values, bootstrap_discount=bootstrap_discounts, priority=priorities)
        
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        # data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        # assert data.discount.shape[1:] == (Config.unroll_length,)
        assert data.discount.shape[0] == Config.unroll_length * Config.num_envs
        chex.assert_shape(data.observation, [Config.unroll_length * Config.num_envs] + list(observation_shape))

        # NOTE we might want to keep the non-flattened data to keep episodes intact 
        # following muzeros further training stuff (re-analyze, GAE etc.)

        buffer_state = replay_buffer.insert(buffer_state, data)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME)
        
        # sampling from replay buffer
        buffer_state, data = replay_buffer.sample(buffer_state)

        # reanalyze
        # data = reanalyze_fn(data, forward_fn, env, reanalyze_key)
        
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step) 
        
        # update replay priorities
        targets = jnp.expand_dims(data.value_prefix_target, -1) + jnp.expand_dims(data.bootstrap_discount, -1) * data.bootstrap_value
        values = forward_fn(data.observation)[1]
        priorities = (jnp.mean(jnp.abs(targets - values), axis=-1) + 1e-10)**Config.per_alpha
        buffer_state = replay_buffer.set_priorities(buffer_state, jnp.squeeze(priorities))
        
        metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
        return (new_training_state, state, buffer_state, new_key), metrics

    def training_epoch(training_state: TrainingState, state: State, buffer_state: ReplayBufferState,
                        key: jnp.ndarray) -> Tuple[TrainingState, State, ReplayBufferState, Metrics]:
        (training_state, state, buffer_state, _), loss_metrics = jax.lax.scan(
            training_step, (training_state, state, buffer_state, key), (),
            length=num_training_steps_per_epoch)
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, buffer_state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: State, buffer_state: ReplayBufferState,
        key: jnp.ndarray) -> Tuple[TrainingState, State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state)) # TODO also needed for replay buffer?
        (training_state, env_state, buffer_state, metrics) = training_epoch(training_state, env_state, buffer_state, key)
        training_state, env_state, metrics = _strip_weak_type((training_state, env_state, metrics))

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (num_training_steps_per_epoch *
            env_step_per_training_step *
            max(Config.num_resets_per_eval, 1)) / epoch_training_time
        current_step = int(_unpmap(training_state.env_steps))
        metrics = {
            'training/total_env_steps': current_step,
            'training/sps': np.round(sps), 
            'training/walltime': np.round(training_walltime), 
            'training/epoch_training_time': np.round(epoch_training_time),
            **{f'training/{name}': float(value) for name, value in metrics.items()} 
        }
        return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
    

    def prefill_replay_buffer(
        training_state: TrainingState, env_state: State,
        buffer_state: ReplayBufferState, key: jnp.ndarray
    ) -> Tuple[TrainingState, State, ReplayBufferState, jnp.ndarray]:

        key_generate_unroll, key_bootstrap, new_key = jax.random.split(key, 3)

        forward_fn = make_forward(
            (training_state.normalizer_params, training_state.params.policy, 
            training_state.params.value, training_state.params.feature_extractor)
        )

        env_state, data = generate_unroll(
            env,
            model_rollout_env,
            env_state,
            forward_fn,
            key_generate_unroll,
            num_prefill_actor_steps,
            deterministic_actions=False,
            extra_fields=('truncation',)
        )

        # additional search at final step for bootstrap values
        _, transition = actor_step(
            env, model_rollout_env, env_state, forward_fn, key_bootstrap, 
            deterministic_actions=False, 
            extra_fields=('truncation',))
        
        assert data.discount.shape[0] == Config.unroll_length

        value_prefix_targets, bootstrap_observations, bootstrap_values, bootstrap_discounts = n_step_fn(
            rewards=data.reward,
            discounts=data.discount * (1 - data.extras['state_extras']['truncation']),
            termination_discount=data.discount,
            observations=data.next_observation,
            values=jnp.concatenate([data.search_value, jnp.array([transition.search_value])]),
        )
        # value_prefix_targets = jnp.zeros_like(value_prefix_targets)
        # bootstrap_discounts = jnp.ones_like(bootstrap_discounts)
        # bootstrap_values = data.search_value

        # NOTE: data.bootstap_value is overloaded with the prior values
        targets = jnp.expand_dims(value_prefix_targets, -1) + jnp.expand_dims(bootstrap_discounts, -1) * bootstrap_values
        priorities = (jnp.mean(jnp.abs(targets - data.bootstrap_value), axis=-1) + 1e-10)**Config.per_alpha

        data = data._replace(value_prefix_target=value_prefix_targets, bootstrap_observation=bootstrap_observations,
                      bootstrap_value=bootstrap_values, bootstrap_discount=bootstrap_discounts, priority=priorities)

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        # data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        # assert data.discount.shape[1:] == (Config.unroll_length,)
        assert data.discount.shape[0] == num_prefill_actor_steps * env_steps_per_actor_step
        chex.assert_shape(data.discount, [num_prefill_actor_steps * env_steps_per_actor_step, ])
        chex.assert_shape(data.observation, [num_prefill_actor_steps * env_steps_per_actor_step] + list(observation_shape))

        # TODO BUFFER probably reshaping / flattening data needed
        # NOTE we might want to keep the current structure nonetheless to keep episodes intact 
        # following muzeros further training stuff (re-analyze etc.)

        buffer_state = replay_buffer.insert(buffer_state, data)

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME)

        new_training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + num_prefill_actor_steps * env_steps_per_actor_step)
        return new_training_state, env_state, buffer_state, new_key
        

    prefill_replay_buffer = jax.pmap(
        prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)


    # initialize params & training state
    if is_atari:
        init_params = AtariAZNetworkParams(
            feature_extractor=az_network.feature_extractor.init(key_feature_extractor),
            policy=az_network.policy_network.init(key_policy),
            value=az_network.value_network.init(key_value))
    else:
        init_params = AtariAZNetworkParams(
            feature_extractor=jnp.zeros(1),
            policy=az_network.policy_network.init(key_policy),
            value=az_network.value_network.init(key_value))
    # else:
    #     init_params = AZNetworkParams(
    #         policy=az_network.policy_network.init(key_policy),
    #         value=az_network.value_network.init(key_value))
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(jnp.zeros(observation_shape)),
        env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])
    
    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    # create eval env
    eval_env = wrap_for_training(
        environment,
        episode_length=Config.episode_length,
        action_repeat=Config.action_repeat,
    )

    evaluator = Evaluator(
        eval_env,
        model_rollout_env,
        make_forward,
        num_eval_envs=Config.num_eval_envs,
        episode_length=Config.episode_length,
        action_repeat=Config.action_repeat,
        key=eval_key,
        deterministic_eval=Config.deterministic_eval,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and eval_env:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (training_state.normalizer_params, training_state.params.policy, 
                 training_state.params.value, training_state.params.feature_extractor)
                ),
            training_metrics={})
        logging.info(metrics)
        # progress_fn(0, metrics)

        # plotting: TEMP, REMOVE AGAIN ###########################################
        # plotting_forwad = make_forward((_unpmap(
        #         (training_state.normalizer_params, training_state.params.policy, 
        #          training_state.params.value, training_state.params.feature_extractor)
        #         )))
        # deterministic_actions = True
        # plotting_key = jax.random.PRNGKey(17)
        # reset_key, unroll_key = jax.random.split(plotting_key)
        # reset_keys = jax.random.split(reset_key, Config.num_eval_envs)
        # plotting_state = eval_env.reset(reset_keys)

        # step_count = 0
        # while plotting_state.done[0] < 1 and plotting_state.info['truncation'][0] < 1:
        #     print(step_count)
        #     search_rng, unroll_key = jax.random.split(unroll_key)
        
        #     prior_logits, value = plotting_forwad(plotting_state.obs)
        #     use_mixed_value = False
        #     root = mctx.RootFnOutput(
        #         prior_logits=prior_logits,
        #         value=value,
        #         embedding=plotting_state, 
        #     )

        #     def recurrent_fn(params, rng_key, action, embedding):
        #         env_state = embedding
        #         nstate = model_rollout_env.step(env_state, action)
        #         prior_logits, value = plotting_forwad(env_state.obs)

        #         recurrent_fn_output = mctx.RecurrentFnOutput(
        #             reward=nstate.reward,
        #             discount=1 - nstate.done,
        #             prior_logits=prior_logits,
        #             value=value,
        #         )
        #         return recurrent_fn_output, nstate

        #     policy_output = mctx.gumbel_muzero_policy(
        #         params=(),
        #         rng_key=search_rng,
        #         root=root,
        #         recurrent_fn=recurrent_fn,
        #         num_simulations=Config.num_simulations,
        #         max_num_considered_actions=Config.max_num_considered_actions,
        #         qtransform=partial(
        #             mctx.qtransform_completed_by_mix_value,
        #             use_mixed_value=use_mixed_value),
        #     )

        #     actions = policy_output.action
        #     action_weights = policy_output.action_weights
        #     best_actions = jnp.argmax(action_weights, axis=-1).astype(jnp.int32)
        #     actions = jax.lax.select(deterministic_actions, best_actions, actions)
            
        #     mctx.draw_tree_to_file(policy_output.search_tree, f'plots/search_tree_00_{step_count}.png')

        #     plotting_state = eval_env.step(plotting_state, actions)
        #     step_count += 1
        # plotting: TEMP, REMOVE AGAIN ###########################################

            

    # prefill replay buffer
    start_prefill = time.time()
    logging.info('prefilling replay buffer')
    if num_prefill_actor_steps > 0:
        prefill_key, local_key = jax.random.split(local_key)
        prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
        training_state, env_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_keys)

    replay_size = jnp.sum(jax.vmap(
        replay_buffer.size)(buffer_state)) * jax.process_count()
    logging.info('replay size after prefill %s, took %s', replay_size, time.time() - start_prefill)
    assert replay_size >= Config.min_replay_size

    # initialize metrics
    training_walltime = 0
    scores = []

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - start_time)

        for _ in range(max(Config.num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, buffer_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, buffer_state, epoch_keys)
            )
            
            logging.info(training_metrics)
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s),
                in_axes=(0, None))(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if Config.num_resets_per_eval > 0 else env_state

        
        
        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.params.policy,
                     training_state.params.value, training_state.params.feature_extractor)),
                training_metrics={})
            
            # record scores
            scores.append((current_step, metrics['eval/episode_reward'], metrics['eval/episode_reward_std'], 
                           metrics['eval/avg_episode_length']))
                
            # eval_metrics = {
            #         'eval/num_episodes': len(eval_returns),
            #         'eval/num_steps': eval_steps,
            #         'eval/mean_score': np.round(np.mean(eval_returns), 3),
            #         'eval/std_score': np.round(np.std(eval_returns), 3),
            #         'eval/mean_episode_length': np.mean(eval_ep_lengths),
            #         'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
            #         'eval/eval_time': eval_time,
            #     }

            logging.info(metrics)
            # progress_fn(current_step, metrics)
            
            # OPTIONAL: save checkpoints of models here
            # params = _unpmap(
            #     (training_state.normalizer_params, training_state.params.policy))

            # plotting: TEMP, REMOVE AGAIN ###########################################
            # plotting_forwad = make_forward((_unpmap(
            #         (training_state.normalizer_params, training_state.params.policy, 
            #         training_state.params.value, training_state.params.feature_extractor)
            #         )))
            # deterministic_actions = True
            # plotting_key = jax.random.PRNGKey(17)
            # reset_key, unroll_key = jax.random.split(plotting_key)
            # reset_keys = jax.random.split(reset_key, Config.num_eval_envs)
            # plotting_state = eval_env.reset(reset_keys)

            # step_count = 0
            # while plotting_state.done[0] < 1 and plotting_state.info['truncation'][0] < 1:
            #     search_rng, unroll_key = jax.random.split(unroll_key)
            
            #     prior_logits, value = plotting_forwad(plotting_state.obs)
            #     use_mixed_value = False
            #     root = mctx.RootFnOutput(
            #         prior_logits=prior_logits,
            #         value=value,
            #         embedding=plotting_state, 
            #     )

            #     def recurrent_fn(params, rng_key, action, embedding):
            #         env_state = embedding
            #         nstate = model_rollout_env.step(env_state, action)
            #         prior_logits, value = plotting_forwad(nstate.obs)

            #         recurrent_fn_output = mctx.RecurrentFnOutput(
            #             reward=nstate.reward,
            #             discount=Config.n_step_gamma * jnp.where(
            #                 nstate.info['truncation'], jnp.ones_like(nstate.done), 1 - nstate.done
            #             ), #1 - nstate.done,
            #             prior_logits=prior_logits,
            #             value=value,
            #         )
            #         return recurrent_fn_output, nstate

            #     # policy_output = mctx.gumbel_muzero_policy(
            #     #     params=(),
            #     #     rng_key=search_rng,
            #     #     root=root,
            #     #     recurrent_fn=recurrent_fn,
            #     #     num_simulations=Config.num_simulations,
            #     #     max_num_considered_actions=Config.max_num_considered_actions,
            #     #     qtransform=partial(
            #     #         mctx.qtransform_completed_by_mix_value,
            #     #         use_mixed_value=use_mixed_value),
            #     # )

            #     policy_output = mctx.sampled_muzero_policy(
            #         params=(),
            #         rng_key=search_rng,
            #         root=root,
            #         recurrent_fn=recurrent_fn,
            #         num_simulations=Config.num_simulations,
            #         dirichlet_fraction=0.25,
            #         dirichlet_alpha=0.3,
            #         pb_c_init=1.25,# 1.25,
            #         pb_c_base=19652, #19652,
            #         temperature=1.0,
            #     )

            #     print_done = (1 - plotting_state.done[0]) * (1 - plotting_state.info['truncation'][0])
            #     # print(step_count, value[0], policy_output.search_tree.summary().value[0], print_done)

            #     actions = policy_output.action
            #     action_weights = policy_output.action_weights
            #     best_actions = jnp.argmax(jnp.mean(policy_output.search_tree.summary().qvalues, axis=-1), axis=-1).astype(jnp.int32)
            #     actions = jax.lax.select(deterministic_actions, best_actions, actions)
                
            #     mctx.draw_tree_to_file(policy_output.search_tree, f'plots/search_tree_{it}_{step_count}.png')

            #     plotting_state = eval_env.step(plotting_state, actions)
            #     step_count += 1
            # plotting: TEMP, REMOVE AGAIN ###########################################


    # END OF TRAINING

    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    # save scores 
    run_dir = os.path.join('experiments', run_name)
    if Config.save_scores:
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        with open(os.path.join(run_dir, "scores.pkl"), "wb") as f:
            pickle.dump(scores, f)

    # if Config.save_model:
    #     model_path = f"weights/{run_name}.params"
    #     with open(model_path, "wb") as f:
    #         f.write(
    #             flax.serialization.to_bytes(
    #                 [
    #                     vars(Config),
    #                     [
    #                         training_state.params.policy,
    #                         training_state.params.value,
    #                         # agent_state.params.feature_extractor,
    #                     ],
    #                 ]
    #             )
    #         )
    #     print(f"model saved to {model_path}")

    # envs.close()
            
    total_steps = current_step
    assert total_steps >= Config.total_timesteps


if __name__ == "__main__":
    app.run(main)