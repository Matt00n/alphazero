# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, Optional, Sequence, TypeVar

import chex
import jax
import jax.numpy as jnp
import pygraphviz


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N, num_atoms]` the raw value for each node.
  node_values: `[B, N, num_atoms]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, Anum_actions` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions, num_atoms]` the value of the next node after the
    action.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N, num_atoms]
  node_values: chex.Array  # [B, N, num_atoms]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  children_visits: chex.Array  # [B, N, num_actions]
  children_rewards: chex.Array  # [B, N, num_actions]
  children_discounts: chex.Array  # [B, N, num_actions]
  children_values: chex.Array  # [B, N, num_actions, num_atoms]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]
  
  @property
  def num_value_atoms(self):
    return self.node_values.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, indices)
    else:
      return _unbatched_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 3)
    value = self.node_values[:, Tree.ROOT_INDEX, :]
    batch_size = value.shape[0]
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 3:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array


def _unbatched_qvalues(tree: Tree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      jnp.expand_dims(tree.children_rewards[index], -1)
      + jnp.expand_dims(tree.children_discounts[index], -1) * tree.children_values[index]
  )


def draw_tree_to_file(
    tree: Tree,
    output_file: str,
    action_labels: Optional[Sequence[str]] = None,
    batch_index: int = 0
) -> pygraphviz.AGraph:
  """Converts a search tree into a Graphviz graph.

  Args:
    tree: A `Tree` containing a batch of search data.
    action_labels: Optional labels for edges, defaults to the action index.
    batch_index: Index of the batch element to plot.

  Returns:
    A Graphviz graph representation of `tree`.
  """
  chex.assert_rank(tree.node_values, 3)
  batch_size = tree.node_values.shape[0]
  if action_labels is None:
    action_labels = range(tree.num_actions)
  elif len(action_labels) != tree.num_actions:
    raise ValueError(
        f"action_labels {action_labels} has the wrong number of actions "
        f"({len(action_labels)}). "
        f"Expecting {tree.num_actions}.")

  def node_to_str(node_i, reward=0, discount=1):
    return (f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {jnp.mean(tree.node_values[batch_index, node_i], axis=-1):.2f}\n" 
            f"Visits: {tree.node_visits[batch_index, node_i]}\n")

  def edge_to_str(node_i, a_i):
    node_index = jnp.full([batch_size], node_i)
    probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
    return (f"{action_labels[a_i]}\n"
            f"Q: {jnp.mean(tree.qvalues(node_index)[batch_index, a_i], axis=-1):.2f}\n"  # pytype: disable=unsupported-operands  # always-use-return-annotations
            f"p: {probs[a_i]:.2f}\n")

  graph = pygraphviz.AGraph(directed=True)

  # Add root
  graph.add_node(0, label=node_to_str(node_i=0), color="green")
  # Add all other nodes and connect them up.
  for node_i in range(tree.num_simulations):
    for a_i in range(tree.num_actions):
      # Index of children, or -1 if not expanded
      children_i = tree.children_index[batch_index, node_i, a_i]
      if children_i >= 0:
        graph.add_node(
            children_i,
            label=node_to_str(
                node_i=children_i,
                reward=tree.children_rewards[batch_index, node_i, a_i],
                discount=tree.children_discounts[batch_index, node_i, a_i]),
            color="red")
        graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

  graph.draw(output_file, prog="dot")
