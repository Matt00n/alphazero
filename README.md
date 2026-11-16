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

* Finishing REANALYZE.
* Enable choosing MCTS version (standard, sampled, Gumbel, stochastic) via Config (has to be manually changed currently).
* Support for continuous action spaces


## MuZero

The working [MuZero](https://arxiv.org/abs/1911.08265) is `muzero.py`. It was consolidated from the step-by-step ablation in `muzero_ablate.py`, which keeps the full history: every change there is flagged with a `### TAG` comment, so `grep "### "` lists them.

### Results on Acrobot-v1

Single seed (22), 1M environment steps. Evaluation uses the most-visited root action, without Dirichlet noise (`eval/*` in `muzero.py`, `eval_visits/*` in `muzero_ablate.py`).

| Setup | Final return (last evaluations) |
|---|---|
| AlphaZero (`alphazero.py`, true environment in search) | −61.5 … −61.9 (spread 0.6–0.9) |
| **MuZero, quantile value head (8 quantiles), 50 simulations** | **−60.5 ± 0.6** |
| MuZero, scalar value head (MSE), 50 simulations | −62.3 ± 1.2 |
| MuZero, scalar head, consistency loss 2.0, 100 simulations, 2M steps | −60.75 ± 1.1 |
| MuZero, categorical head in raw space (81 atoms) | −64.3 ± 3.3 |
| MuZero, MuZero's categorical h-space head (81 / 201 / 601 atoms) | unstable, −70 … −500 |

The optimum is around −60. A second seed for the quantile head is pending.

### Recommended settings

These are the defaults in `muzero.py`, unless the table says otherwise.

| Group | Setting | Value | Note |
|---|---|---|---|
| Value head | `categorical_value`, `num_atoms`, `qr_kappa` | `False`, `8`, `1.0` | Quantile regression on raw returns: no transform bias, no binning. `num_atoms=1` gives a scalar MSE head. |
| Reward head | `reward_support_max` | `None` | MuZero's 601-atom categorical reward head. |
| Value targets | `use_gae`, `gae_lambda`, `n_step_gamma` | `True`, `0.95`, `0.99` | GAE over the *stored* MCTS search values, as in AlphaZero. Do not bootstrap from the online network at loss time. |
| Model | `learned_dynamics_search` | `True` | Full MuZero. `False` runs the search on the true environment (oracle ablation). |
| Unrolled loss | `loss_unroll_length` | `5` | |
| Unrolled loss | `dynamics_gradient_scale` | `0.5` | MuZero's gradient scaling at the dynamics input. |
| Unrolled loss | per-step loss weights | always on | Weight 1 at k=0 and 1/K for each k≥1, as in MuZero. |
| Unrolled loss | `consistency_loss_coef` | `0.0` | Not in the original MuZero. 2.0 (EfficientZero) helped the scalar head bootstrap before reward-aware PER existed. |
| Search | `num_simulations` | `50` | 100 is slightly better and twice as slow. |
| Search | Dirichlet α, fraction; `pb_c_init`, `pb_c_base` | 0.3, 0.25; 1.25, 19652 | MuZero's values. |
| Replay | `max_replay_size`, `min_replay_size`, `replay_buffer_batch_size` | 32768, 8192, 128 | 8192 was too small. A batch of 256 made no difference. |
| Replay | `per_alpha`, `per_beta_anneal` (0.4 → 1.0), `per_epsilon` | 0.6, `True`, 1e-6 | EfficientZero's values. |
| Replay | `per_reward_priority_weight` | `1.0` | Priority = value error + reward error. This is what lets the agent bootstrap from rare goal transitions. |
| Optimiser | `learning_rate`, `late_lr_decay`, `lr_decay_start_frac` | 1e-3, `True`, 0.8 | Constant, then linear decay to 1% over the last 20% of updates. |
| Optimiser | `max_grad_norm` | `50` | Effectively a safety net. 0.5 clipped almost every update. |
| Optimiser | `l2_coef` | 1e-4 | L2 on kernels only. L2 on LayerNorm gains collapsed the representation. |
| Networks | embedding; representation, dynamics, value, policy layers | 64; 64×2, 64×2, 256×5, 32×4 (swish) | Larger networks bootstrapped much later. |
| Data | `num_envs`, `unroll_length`, `total_timesteps` | 16, 128, 1e6 | |
| Evaluation | `eval_argmax_q` | `False` | `eval/*` uses the most-visited action without noise. `True` adds an argmax-Q evaluator with noise (`eval_argmax_q/*`). |

### Notes on tuning

**Running experiments**
- Runs are deterministic for a given seed and config. Compare variants on the same seed.
- `eval_every` changes the epoch structure and the random streams, so keep it fixed when comparing.
- Judge stability on the visit-count evaluation (`eval/*`). Root Dirichlet noise in the argmax-Q evaluator makes good agents look unstable: one run read −74.8 ± 56 there and −62.5 ± 1.4 with visit counts.

**Learning-rate schedules**
- optax schedules count optimiser updates, not environment steps. A 1M-step run has only about 68k updates, so `transition_steps=total_timesteps` barely decays the learning rate.
- Annealing all the way down can freeze a suboptimal strategy. On Acrobot there are two attractors, around −62 and −83. Decay late instead (`late_lr_decay`).

**Value heads**
- MuZero's h-space categorical head estimates h⁻¹(E[h(G)]). Under noisy targets this is biased towards smaller |V| (Jensen's inequality), and more atoms do not remove the bias.
- On returns of order 100, prefer the quantile head, the scalar head, or the raw-space two-hot (`value_raw_space_twohot`).
- `value_abs_err_k0` tracks how accurate the decoded values are.

**Bootstrapping from rare rewards**
- Under random play there is about one goal per 20k steps. If the reward loss sits at its cross-entropy floor and `goal*` metrics stay near zero, the model has not seen the goal yet.
- Reward-aware PER fixes this. Larger networks and larger buffers make it worse.

**Representation collapse**
- Watch the collapse metrics: pairwise cosine, feature std and LayerNorm gain.
- Watch the gradient norms and `grad_clip_frac`.

**Model vs. planning**
- `planning_diagnostics=True` compares the learned-model search with a true-environment search from the same states. It reports agreement, regret, Q errors and value drift for k = 1…8, and roughly doubles evaluation time.
- A longer unroll (9) made the model more accurate but the agent worse. With a 64-dim embedding, the extra model accuracy cost policy and value quality.

**Time limits**
- Truncation must not be treated as termination. Time-limit steps are bootstrapped in GAE, and unroll steps after a truncation are masked rather than made absorbing.
- The n-step path (`use_gae=False`) still treats truncation as terminal.

### What it took, start to finish

**Bugs fixed:**
1. **Value targets** were bootstrapped from the current network at loss time. They now use stored GAE over the MCTS search values.
2. **Prioritised replay read and wrote the wrong columns** of `MuZeroTransition`: goal transitions were never sampled, and `terminal_mask` was overwritten.
3. **Truncation was treated as termination,** both in value targets and in the model's absorbing states.
4. **Smaller fixes:** the n-step padding, the categorical decoding, PER importance weights (now normalised per batch), and the learning-rate schedule counting updates rather than environment steps.

**Changes that made the difference:**
- **Scalar, and later quantile, value head** instead of 601 atoms.
- **MuZero's gradient scaling and per-step loss weights.**
- **Reward-aware PER** (or, before that, consistency loss 2.0).
- **50 simulations.**
- **Buffer 32768.**
- **Gradient clip at 50, and no L2 on biases or LayerNorm.**
- **Learning-rate decay over the last 20%.**
- **Evaluation by visit count.**
