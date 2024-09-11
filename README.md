# AlphaZero Reimplementation

This is a reimplementation of the infamous [AlphaZero](https://arxiv.org/abs/1712.01815) algorithm in JAX. At its core, we use [MCTX](https://github.com/google-deepmind/mctx) for the tree search. The system design is inspired by [BRAX](https://github.com/google/brax).

### Main features

* Performance on par with the original AlphaZero on common RL environments.
* Fast training and tree search via [JAX](https://jax.readthedocs.io/en/latest/) and [MCTX](https://github.com/google-deepmind/mctx).
* Prioritized replay buffer.
* Support for sample-based versions of AlphaZero as in [Sampled MuZero](https://arxiv.org/abs/2104.06303). The different variants of AlphaZero (standard, sampled, Gumbel) can easily be switched in.
* Option for using a distributional value function via [Quantile Regression](https://arxiv.org/abs/1710.10044) similarly as in [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4).
* Option for using [GAE (Generalized Advantage Estimation)](https://arxiv.org/abs/1506.02438) instead of n-step returns.
* Environment support for [gymnax](https://github.com/RobertTLange/gymnax). Minimal modification of the environments is required, this has been done for CartPole, MountainCar and Acrobot so far. Other common JAX environment libraries can also be plugged in with small adaptations.
* Option for [REANALYZE](https://arxiv.org/abs/2104.06294). *This is work in progress and not fully implemented yet!*.

### Potential future extensions (if I have time...)

* Extension to [MuZero](https://arxiv.org/abs/1911.08265) by learning dynamics model.
* Finishing REANALYZE.
* Enable choosing MCTS version (standard, sampled, Gumbel, stochastic) via Config (has to be manually changed currently).
* Support for continuous action spaces





