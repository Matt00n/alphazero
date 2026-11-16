from collections.abc import Mapping
from typing import Any, Protocol, TypeVar

import jax.numpy as jnp

NetworkType = TypeVar('NetworkType')


class Policy(Protocol):

  def __call__(
      self,
      observation: jnp.ndarray,
      key: jnp.ndarray,
  ) -> tuple[jnp.ndarray, Mapping[str, Any]]:
    pass


class ForwardPass(Protocol):

  def __call__(
      self,
      observation: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    pass


class BaseRepresentationFn(Protocol):

  def __call__(
      self,
      observation: jnp.ndarray,
  ) -> jnp.ndarray:
    pass


class BaseDynamicsFn(Protocol):

  def __call__(
      self,
      hidden_state: jnp.ndarray,
      action: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    pass
  

class NetworkFactory(Protocol[NetworkType]):

  def __call__(
      self,
      observation_size: int,
      action_size: int,
      preprocess_observations_fn,
  ) -> NetworkType:
    pass