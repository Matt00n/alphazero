"""Muzero reimplementation."""

# DONE: Compatibility with Jax environment
# DONE: Port to Gymnax (convert Gymnax API to Brax (done), modify gymnax to return truncated flags (done), env returns only jax arrays (done))
# DONE ENV: Ensure that scores are maintained 
# DONE ENV: dissolve all dependency issues in subfiles
# (TODO ENV: atari wrappers, atari networks, change MinAtar envs)
# TODO ENV: visualizer
# (TODO ENV: Option for passing env params to all functions which step env)
# DONE Blueprint of all Muzero elements
# DONE MCTS: Adapt policy learning
# TODO MCTS: Learn Model
# TODO MCTS: Replay Buffers from Brax (logic for prefilling buffer before training)
# TODO MCTS: keep options for both replay & rollout buffer --> test to see what works better in target use case
# TODO MCTS: option for different MCTS versions from MCTX
# NOT NEEDED (handled internally by mcts via discounts): Auto reset not wanted in MCTS rollouts
# TODO: System adapted from EfficientZero
# TODO: Adapt implementation details from further papers: EfficientZero, Reanalyze, Sampled Muzero, Gumble Muzero, AlphaTensor, AlphaDev, DreamerV3
# TODO: look into running jax on (M1) GPU: Gymnax, Jax docs, Jax M1
# TODO: clean up and organize code / folders
# DONE Test with Gymnax environments
# TODO pass in mcts policy as argument so that we can partial it before
# TODO make eval deterministic (as option) for non gumbel version
# TODO potentially change generate unroll to generate episodes (remove auto reset and reset manually after unroll)

# Optional / future research
# TODO incorporate GAE into tree search to construct targets
# TODO Distributional RL incorporated in MCTS
# TODO MCTS replay vs rollout buffer



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
from envs.brax_wrappers import EvalWrapper, wrap_for_training
from networks.policy import Policy, ForwardPass
from networks.networks import FeedForwardNetwork, ActivationFn, make_policy_network, make_value_network, make_atari_feature_extractor
from networks.distributions import NormalTanhDistribution, ParametricDistribution, PolicyNormalDistribution, DiscreteDistribution
import replay_buffers
import running_statistics
from gymnax import gymnax
from gymnax.gymnax.wrappers.brax import GymnaxToBraxWrapper, State
import mctx

class Config:
    # TODO BUFFER update configs

    # experiment
    experiment_name = 'ppo_test'
    seed = 30
    platform = 'cpu' # CPU or GPU
    capture_video = False
    write_logs_to_file = False
    save_model = False
    save_scores = False

    # environment
    env_id = 'CartPole-v1' # CartPole-v1, Breakout-MinAtar
    num_envs = 16
    normalize_observations = True 
    action_repeat = 1
    eval_env = True
    num_resets_per_eval = 0
    eval_every = 1
    deterministic_eval = True
    num_eval_envs = 64
    episode_length = 1000

    # MCTS
    num_simulations = 30
    max_num_considered_actions = 16
    l2_coef = 1e-4
    vf_cost = 0.5

    # algorithm hyperparameters
    total_timesteps = int(1e6) 
    learning_rate = 1e-3 # 3e-4 
    unroll_length = 128 # 128 
    anneal_lr = True
    batch_size = 1
    num_minibatches = 64
    update_epochs = 1 # 10
    max_grad_norm = 0.5
    grad_updates_per_step: int = 1 # replaces former update_epochs
    
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
class PPONetworkParams:
    """Contains training state for the learner."""
    policy: Any
    value: Any


@flax.struct.dataclass
class PPONetworks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class AtariPPONetworkParams:
    """Contains training state for the learner."""
    feature_extractor: Any
    policy: Any
    value: Any


@flax.struct.dataclass
class AtariPPONetworks:
    feature_extractor: FeedForwardNetwork
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Union[PPONetworkParams, AtariPPONetworkParams]
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def make_inference_fn(ppo_networks: Union[PPONetworks, AtariPPONetworks]):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: Any,
                    deterministic: bool = False,
                    use_feature_extractor: bool = False,
                    ) -> Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution
        if use_feature_extractor:
            shared_feature_extractor = ppo_networks.feature_extractor
        normalizer_params, policy_params, feature_extractor_params = params

        @jax.jit # TODO jit needed ?
        def policy(observations: jnp.ndarray,
                key_sample: jnp.ndarray) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            if use_feature_extractor:
                observations = shared_feature_extractor.apply(normalizer_params, feature_extractor_params, observations)
            logits = policy_network.apply(normalizer_params, policy_params, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
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


def make_forward_fn(ppo_networks: Union[PPONetworks, AtariPPONetworks]):
    """Creates params and inference function for the PPO agent."""

    def make_forward(params: Any,
                    use_feature_extractor: bool = False,
                    ) -> ForwardPass:
        policy_network = ppo_networks.policy_network
        value_network = ppo_networks.value_network
        if use_feature_extractor:
            shared_feature_extractor = ppo_networks.feature_extractor
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


# def make_feature_extraction_fn(ppo_networks: AtariPPONetworks):
#     """Creates feature extractor for inference."""

#     def make_feature_extractor(params: Any):
#         shared_feature_extractor = ppo_networks.feature_extractor

#         @jax.jit
#         def feature_extractor(observations: jnp.ndarray) -> jnp.ndarray:
#             return shared_feature_extractor.apply(*params, observations)

#         return feature_extractor

#     return make_feature_extractor


def make_ppo_networks(
        observation_size: Union[Sequence[int], int],
        action_size: int,
        preprocess_observation_fn: Callable, 
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: ActivationFn = nn.swish,
        sqash_distribution: bool = True,
        discrete_policy: bool = False,
        shared_feature_extractor: bool = False,
        feature_extractor_dense_hidden_layer_sizes: Optional[Sequence[int]] = (512,),
    ) -> Union[PPONetworks, AtariPPONetworkParams]:
    """Make PPO networks with preprocessor."""
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
            hidden_layer_sizes=(),
            activation=activation)
        return AtariPPONetworks(
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
        preprocess_observation_fn=preprocess_observation_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation)

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


def actor_step(
    env: GymnaxToBraxWrapper,
    env_state: State,
    forward: ForwardPass,
    key: jnp.ndarray,
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
        nstate = env.step(env_state, action)

        # policy & value networks
        prior_logits, value = forward(env_state.obs)

        # Create the new MCTS node.
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=nstate.reward,
            # discount when terminal state reached
            discount=1 - nstate.done,
            # prior for the new state
            prior_logits=prior_logits,
            # value for the new state
            value=value,
        )

        # Return the new node and the new environment.
        return recurrent_fn_output, nstate

    # Running the search.
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=search_rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=Config.num_simulations,
        max_num_considered_actions=Config.max_num_considered_actions,
        qtransform=partial(
            mctx.qtransform_completed_by_mix_value,
            use_mixed_value=use_mixed_value),
    )

    actions = policy_output.action
    ##################################### end mctx ######################################
    
    search_value = policy_output.search_tree.summary().value

    policy_extras = {
        'prior_log_prob': tfd.Categorical(logits=prior_logits).log_prob(actions),
        'raw_action': actions
    }

    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, MCTSTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        target_policy_probs=policy_output.action_weights,
        target_value=search_value,
        extras={
            'policy_extras': policy_extras, 
            'state_extras': state_extras
        })


def generate_unroll(
    env: GymnaxToBraxWrapper,
    env_state: State,
    forward: ForwardPass,
    key: jnp.ndarray,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, MCTSTransition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(
            env, state, forward, current_key, extra_fields=extra_fields)
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length)
    return final_state, data


class Evaluator:
    """Class to run evaluations."""

    def __init__(self, eval_env: GymnaxToBraxWrapper,
                eval_forward_fn: Callable[[Any],
                                            Policy], num_eval_envs: int,
                episode_length: int, action_repeat: int, key: jnp.ndarray):
        """Init.

        Args:
            eval_env: Batched environment to run evals on.
            eval_policy_fn: Function returning the policy from the policy parameters.
            num_eval_envs: Each env will run 1 episode in parallel for each eval.
            episode_length: Maximum length of an episode.
            action_repeat: Number of physics steps per env step.
            key: RNG key.
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
                eval_first_state,
                eval_forward_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat)[0]

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


def compute_muzero_loss(
    params: Union[PPONetworkParams, AtariPPONetworkParams],
    normalizer_params: Any,
    data: MCTSTransition,
    rng: jnp.ndarray,
    ppo_network: Union[PPONetworks, AtariPPONetworks],
    vf_cost: float = 0.5,
    l2_coef: float = 1e-4,
    shared_feature_extractor: bool = False,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes Alphazero loss .

    Args:
        params: Network parameters.
        normalizer_params: Parameters of the normalizer.
        data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
            ['policy_extras']['log_prob']
        rng: Random key
        ppo_network: PPO networks.
        vf_cost: Value loss coefficient.
        l2_coef: L2 penalty coefficient.
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    if shared_feature_extractor:
        feature_extractor_apply = ppo_network.feature_extractor.apply
        hidden = feature_extractor_apply(normalizer_params, params.feature_extractor, data.observation)
    
    policy_logits = policy_apply(normalizer_params, params.policy,
                                hidden)

    baseline = value_apply(normalizer_params, params.value, hidden)

    policy_targets = data.target_policy_probs

    # target_action_log_probs = parametric_action_distribution.log_prob(
    #     policy_logits, data.extras['policy_extras']['raw_action'])
    # behaviour_action_log_probs = data.extras['policy_extras']['prior_log_prob']

    # log_ratio = target_action_log_probs - behaviour_action_log_probs
    # rho_s = jnp.exp(log_ratio)
    # approx_kl = ((rho_s - 1) - log_ratio).mean()

    policy_loss = -jnp.mean(jnp.sum(jax.lax.stop_gradient(policy_targets)*(jax.nn.log_softmax(policy_logits)), axis=-1))

    # Value function loss
    vs = data.target_value
    v_error = jax.lax.stop_gradient(vs) - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_cost

    # l2 penalty
    l2_penalty = l2_coef * 0.5 * sum(
        jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(params))

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))

    total_loss = policy_loss + v_loss + l2_penalty

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': v_loss,
        'l2_penalty': l2_penalty,
        'entropy': entropy, 
        # 'approx_kl': jax.lax.stop_gradient(approx_kl), 
    }

    return total_loss, metrics





def main(_):
    start_time = time.time()

    run_name = f"Exp_{Config.experiment_name}__{Config.env_id}__{Config.seed}__{int(time.time())}"

    if Config.write_logs_to_file:
        from absl import flags
        flags.FLAGS.alsologtostderr = True
        log_path = f'./training_logs/ppo/{run_name}'
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

    
    assert Config.batch_size * Config.num_minibatches % Config.num_envs == 0
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        Config.batch_size * Config.unroll_length * Config.num_minibatches * Config.action_repeat) 
    num_training_steps = np.ceil(Config.total_timesteps / env_step_per_training_step).astype(int)
    num_evals_after_init = max(np.floor(num_training_steps / Config.eval_every).astype(int), 1)
    num_training_steps_per_epoch = np.ceil(
        Config.total_timesteps / (num_evals_after_init * env_step_per_training_step * max(Config.num_resets_per_eval, 1))
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

    # Normalize moved from env to network
    normalize = lambda x, y: x
    if Config.normalize_observations:
        normalize = running_statistics.normalize

    ppo_network = make_ppo_networks(
        observation_size=observation_shape, # NOTE only works with flattened observation space
        action_size=action_size, # flatten action size for nested spaces
        preprocess_observation_fn=normalize, 
        policy_hidden_layer_sizes=Config.policy_hidden_layer_sizes, 
        value_hidden_layer_sizes=Config.value_hidden_layer_sizes,
        activation=Config.activation,
        sqash_distribution=Config.squash_distribution,
        discrete_policy=discrete_action_space,
        shared_feature_extractor=is_atari,
        feature_extractor_dense_hidden_layer_sizes=Config.atari_dense_layer_sizes,
    )
    make_forward = make_forward_fn(ppo_network)
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

    # create loss function via functools.partial
    loss_fn = partial(
        compute_muzero_loss,
        ppo_network=ppo_network,
        vf_cost=Config.vf_cost,
        l2_coef=Config.l2_coef,
        shared_feature_extractor=is_atari,
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
            x = jax.random.permutation(key_perm, x)
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
        carry: Tuple[TrainingState, State, jnp.ndarray],
        unused_t
    ) -> Tuple[Tuple[TrainingState, State, jnp.ndarray], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        forward_fn = make_forward(
            (training_state.normalizer_params, training_state.params.policy, 
             training_state.params.value, training_state.params.feature_extractor)
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = generate_unroll(
                env,
                current_state,
                forward_fn,
                current_key,
                Config.unroll_length,
                extra_fields=('truncation',))
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f, (state, key_generate_unroll), (),
            length=Config.batch_size * Config.num_minibatches // Config.num_envs) 
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        assert data.discount.shape[1:] == (Config.unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME)
        
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step) 
        return (new_training_state, state, new_key), metrics


    def training_epoch(training_state: TrainingState, state: State,
                        key: jnp.ndarray) -> Tuple[TrainingState, State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step, (training_state, state, key), (),
            length=num_training_steps_per_epoch)
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: State,
        key: jnp.ndarray) -> Tuple[TrainingState, State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

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
        return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade
    

    # initialize params & training state
    if is_atari:
        init_params = AtariPPONetworkParams(
            feature_extractor=ppo_network.feature_extractor.init(key_feature_extractor),
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    else:
        init_params = AtariPPONetworkParams(
            feature_extractor=jnp.zeros(1),
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    # else:
    #     init_params = PPONetworkParams(
    #         policy=ppo_network.policy_network.init(key_policy),
    #         value=ppo_network.value_network.init(key_value))
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(jnp.zeros(observation_shape)),
        env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])
    
    
    # create eval env
    eval_env = wrap_for_training(
        environment,
        episode_length=Config.episode_length,
        action_repeat=Config.action_repeat,
    )

    evaluator = Evaluator(
        eval_env,
        make_forward,
        num_eval_envs=Config.num_eval_envs,
        episode_length=Config.episode_length,
        action_repeat=Config.action_repeat,
        key=eval_key,
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


    # initialize metrics
    training_walltime = 0
    scores = []

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - start_time)

        for _ in range(max(Config.num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, epoch_keys)
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