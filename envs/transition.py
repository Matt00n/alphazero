from typing import NamedTuple

import jax.numpy as jnp

NestedArray = jnp.ndarray

class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    real_obs: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class MCTSTransition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    real_obs: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    target_policy_probs: NestedArray
    search_value: NestedArray
    value_prefix_target: NestedArray
    bootstrap_observation: NestedArray
    bootstrap_value: NestedArray
    bootstrap_discount: NestedArray
    extras: NestedArray  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    priority: NestedArray
    weight: NestedArray


class MuZeroTransition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    real_obs: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    target_policy_probs: NestedArray
    search_value: NestedArray
    value_prefix_target: NestedArray
    bootstrap_observation: NestedArray
    bootstrap_value: NestedArray
    bootstrap_discount: NestedArray
    extras: NestedArray  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    priority: NestedArray
    weight: NestedArray
    
    # unroll targets 
    unroll_obs: NestedArray
    policy_targets: NestedArray
    value_prefix_targets: NestedArray
    bootstrap_discounts: NestedArray
    bootstrap_values: NestedArray
    bootstrap_observations: NestedArray
    reward_targets: NestedArray
    unroll_actions: NestedArray
    unroll_mask: NestedArray
