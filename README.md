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
* Partial observability and memory in `muzero.py`: observation subsets (`observation_indices`), MuZero-style history stacking (`history_length`), and a GRU representation over the history (`representation_type="gru"`), optionally with its state carried through the episode (`rnn_carry_state`, R2D2-style). See [Partial observability](#partial-observability-and-memory).
* Option for [MuZero Reanalyze](https://arxiv.org/abs/2104.06294) in `muzero.py` (`Config.reanalyze`): stored trajectories are searched again with the current networks and their targets are refreshed in the replay buffer. On Acrobot it gave no consistent gain (see below).

### Potential future extensions (if I have time...)

* Enable choosing MCTS version (standard, sampled, Gumbel, stochastic) via Config (has to be manually changed currently).
* Support for continuous action spaces


## MuZero

The working [MuZero](https://arxiv.org/abs/1911.08265) is `muzero.py`. It was consolidated from the step-by-step ablation in `muzero_ablate.py`, which keeps the full history: every change there is flagged with a `### TAG` comment, so `grep "### "` lists them.

### Results on Acrobot-v1

Seed 22 unless noted, 1M environment steps. Evaluation uses the most-visited root action, without Dirichlet noise (`eval/*` in `muzero.py`, `eval_visits/*` in `muzero_ablate.py`).

| Setup | Final return (last evaluations) |
|---|---|
| AlphaZero (`alphazero.py`, true environment in search) | −61.5 … −61.9 (spread 0.6–0.9) |
| **MuZero, scalar value head (MSE), 50 simulations (default)** | **−62.3 ± 1.2; seed 23: −62.1 ± 0.4** |
| MuZero, scalar head + reanalyze | −63.0 ± 2.1 |
| MuZero, quantile value head (8 quantiles), 50 simulations | −60.5 ± 0.6; seed 23: −82.7 ± 1.7 |
| MuZero, quantile head + reanalyze | −82.2 ± 2.5; seed 23: −61.0 ± 0.4 |
| MuZero, scalar head, consistency loss 2.0, 100 simulations, 2M steps | −60.75 ± 1.1 |
| MuZero, categorical head in raw space (81 atoms) | −64.3 ± 3.3 |
| MuZero, MuZero's categorical h-space head (81 / 201 / 601 atoms) | unstable, −70 … −500 |

The optimum is around −60. Runs end on one of two strategies, about −62 or about −83 (one extra swing). The scalar head reached −62 in all three of its runs. The quantile head gave the best single run but ended at −83 in two of four. Reanalyze flipped the quantile head's outcome on both seeds, in opposite directions, so the choice of strategy looks like chance per run rather than an effect of either option.

**The quantile head is worth keeping.** It gave the two best runs, −60.5 and −61.0, both better than AlphaZero, with the lowest end-of-run spread (±0.4–0.6). Its weak point is not value accuracy: in the −83 runs the value net was accurate (see [Strategy lock-in](#notes-on-tuning)). So anything that makes runs reliably pick the better strategy should benefit the quantile head most. Then it would be the natural default. Try it with `num_atoms=8` and compare on several seeds.

### Recommended settings

These are the defaults in `muzero.py`, unless the table says otherwise.

| Group | Setting | Value | Note |
|---|---|---|---|
| Value head | `categorical_value`, `num_atoms`, `qr_kappa` | `False`, `1`, `1.0` | Scalar MSE head on raw returns: no transform bias, no binning. `num_atoms=8` gives the quantile head: best single run, less robust. |
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
| Reanalyze | `reanalyze`, `reanalyze_windows_per_step` | `False`, 1 | No consistent gain on Acrobot, 1.3–1.9× training time (see below). |
| Partial observability | `observation_indices`, `history_length`, `history_actions`, `representation_type`, `rnn_hidden_size`, `rnn_carry_state` | `None`, 1, `True`, `"mlp"`, 64, `False` | Off by default. See [Partial observability](#partial-observability-and-memory). |
| Evaluation | `eval_argmax_q` | `False` | `eval/*` uses the most-visited action without noise. `True` adds an argmax-Q evaluator with noise (`eval_argmax_q/*`). |

### Reanalyze

With `reanalyze=True`, every training step draws `reanalyze_windows_per_step` random windows of stored transitions from the replay buffer. A window is `unroll_length × num_envs` consecutive steps, the same size as one acting unroll. Each window is searched again with the current networks, with root noise as when acting (`reanalyze_dirichlet_fraction`). Its policy targets, search values (and so its GAE value targets), unroll targets and replay priorities are then recomputed exactly as for fresh data, and written back.

- **Why this works:** the buffer holds one continuous, time-major stream of the environments. Any aligned window is a block of consecutive transitions, and the rows right after it are the states to bootstrap from. No environment state has to be stored, because the search only needs observations.
- **Cost:** each window costs as much search as one acting unroll, so one window per step roughly doubles search time.
- **Requirements:** `learned_dynamics_search=True`, `num_resets_per_eval=0`, and a buffer size that is a multiple of `num_envs`.
- **Metrics:**
  - `reanalyze_policy_tv`: total variation between the new and the stored policy targets.
  - `reanalyze_value_target_change`: mean absolute change of the k=0 value targets.
  - `reanalyze_window_age`: how many training steps ago the window was collected.
- **Results on Acrobot (1 window per step):** no consistent effect on the final return (see the table above). Goals were found and the policy bootstrapped later, by about 100k–250k steps, and training took 1.3–1.9× as long. The policy targets change a lot early on (`reanalyze_policy_tv` about 0.2–0.3) and little by the end (about 0.07). Acrobot is cheap to simulate, so fresh data is a better use of the compute here. Reanalyze is meant for settings where environment steps are expensive.
- **Differences from the paper:** value targets stay GAE over search values, not n-step returns from a target network. There is no separate fraction of reanalysed targets: refreshed windows simply replace their old targets in the buffer.

### Partial observability and memory

MuZero's representation takes a history of observations and actions (32 frames on Atari), not just the current observation. `muzero.py` supports this through environment wrappers in `envs/pomdp_wrappers.py`, plus a choice of representation network:

- `observation_indices`: keep only these observation entries, to make an environment partially observable. For Acrobot, `(0, 1, 2, 3)` drops the two angular velocities.
- `history_length` (H > 1): the observation becomes the last H frames, oldest first. Each frame is `[observation, one-hot of the previous action, valid flag]`, and the valid flag is 0 for padding before the start of the episode. `history_actions=False` leaves out the action.
- `representation_type`:
  - `"mlp"` (default): the usual representation MLP on the flat input. With H > 1 this is MuZero's history stacking.
  - `"gru"`: a frame encoder (`representation_hidden_layer_sizes`), then a GRU (`rnn_hidden_size`) over the H frames from a zero state, then a LayerNorm head. Padding frames are skipped.
- `rnn_carry_state=True` (with `"gru"`): the GRU state is carried through the whole episode, so memory reaches beyond the window. H=1 is allowed here.
- With a history, only the observation entries of each frame are normalised. The action one-hot, the valid flag and the carried GRU state stay raw. The valid flag is rarely 0, so its running standard deviation is tiny and normalising it gave values around −13 at episode starts.

Design notes:
- **The history lives in the observation.** The wrappers sit directly on the gymnax environment, below the episode and auto-reset wrappers, so an auto-reset also resets the history. The replay buffer, the unrolled loss, reanalyze, the true-environment search and the evaluators all work unchanged.
- **Without carrying, the GRU sees only the window.** It is recomputed from the H stored frames each time, so acting, training and reanalyze compute exactly the same embedding, and there is no stale recurrent state. Memory is limited to H steps.
- **With carrying, the stored state is the one from before the window (R2D2-style).** Each transition stores `carry0`, the GRU state from just before the oldest frame of its window, and the observation becomes `[carry0, H frames]`. The representation runs the GRU from `carry0` through the window. The window acts as R2D2's burn-in, but gradients flow through it (truncated backpropagation through time over H steps).
  - The next step's `carry0` is one GRU step on the oldest frame. It is reset to zero when an episode ends.
  - The agent keeps `carry0` in `env_state.info["rnn_carry"]` (`RecurrentStateSlotWrapper`), so acting, prefill and evaluation carry it without extra plumbing.
  - The buffer, loss and reanalyze are unchanged. `carry0` is stale in the usual R2D2 sense: it was computed with the network weights at acting time. Reanalyze refreshes the targets but not `carry0`.
  - The true-environment search (`learned_dynamics_search=False`) and `planning_diagnostics` are not supported with carrying: they embed environment states without the agent's recurrent state.
- **Learned dynamics need no memory of their own.** Once the representation has encoded the history, the hidden state is all that the dynamics, value and policy use, as in MuZero.

Results on Acrobot without velocities (`observation_indices=(0, 1, 2, 3)`; scalar head, 1M steps, H=4, with the normalisation above). "First solved" is the first evaluation that reaches the goal in every episode:

| Setup | Seed 22 | Seed 23 |
|---|---|---|
| Full observation (reference) | −62.3 ± 1.2 | −62.1 ± 0.4 |
| Stacking (`history_length=4`) | −108 ± 31, first solved at 910k, still improving | **−66 ± 6**, first solved at 350k, still improving |
| GRU over the window (`representation_type="gru"`) | −500: never learned to reach the goal | −74 ± 8, first solved at 350k |
| Carried GRU (`rnn_carry_state=True`) | −195 … −280 ± 110–120: reaches the goal in only some episodes | −91 … −105 ± 25, first solved at 1M |

Before the normalisation change (seed 22 only): no memory −500, stacking −500, GRU −82 … −87.

- **Memory is necessary.** Without it the agent never learns to reach the goal.
- **Finding the goal is the bottleneck, not the memory architecture.** Every run first sits on a plateau. Almost no sampled batch contains a goal (0–5%), and the value net fits the constant failure return almost perfectly (value loss down to 0.001), so there is nothing to learn from. Once reward-aware PER fills the batches with goals, the policy is solved within about 50k–100k steps. When that happens varies by seed: 350k to past 1M, against about 140k with full observations. So 1M steps is too short to compare final returns. The normalisation change also re-randomises each run: stacking on seed 22 went from never solving to −108, while the GRU on seed 22 went the other way.
- **Stacking is the simplest choice here, and the best so far.** Four frames of angles are enough to infer the velocities, so Acrobot without velocities needs only short memory.
- **Carrying the GRU state has not helped on this task.** It was slower to solve on both seeds and unstable on seed 22. Acrobot gives long memory nothing extra to use, and the stored `carry0` is stale (see above). A task with long-range dependencies would be needed to show its benefit.

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
- On returns of order 100, prefer the scalar head, the quantile head, or the raw-space two-hot (`value_raw_space_twohot`).
- `value_abs_err_k0` tracks how accurate the decoded values are.
- For the quantile head, keep `qr_kappa=1`. κ=10 was worse on both seeds: −61.6 instead of −60.5 on seed 22, and −94 instead of −83 on seed 23. Its value error was higher throughout.

**Strategy lock-in (−62 vs −83)**
- Diagnosis (seed-23 quantile run, final checkpoint at −83 vs the reanalyze run's checkpoint at −61, true-environment rollouts from every state of the −83 episodes):
  - No single-action change helps: changing only the current action and then following the −83 policy gives returns within about 1 of each other at all 338 states.
  - A multi-step change helps a lot: from about step 12–14, the −61 policy's continuation is about 11 better in discounted return (about 20 steps shorter).
  - The value net is accurate for its own policy, and the learned-model search agrees with a true-environment search. Neither value accuracy nor the model is the problem.
  - The prior is collapsed (0.96–0.99 on one action). With 50 or 400 simulations, and even on the true environment, the search visits only that action.
- So −83 is a local optimum of one-step policy improvement, which is what MCTS-based policy iteration does. The strategy is chosen while the policy sharpens: in the −83 run, search entropy fell from about 0.95 to 0.3 between 350k and 520k steps, when goals started to appear. Root Dirichlet noise and sampling from visit counts only try single-step deviations, and those look no better.
- Options to make runs pick the better strategy more reliably, none tested yet. The aim of all of them is to keep multi-step alternatives alive until the values can tell them apart:
  - *Existing knobs:* a larger Dirichlet fraction or α at the root (0.25 and 0.3, set in `run_search`); larger `pb_c_init` (1.25) for more exploration inside the tree; `late_lr_decay` later or off (annealing freezes whichever strategy is current); more `num_simulations` in training.
  - *Needs code:* an entropy bonus on the policy loss, or a floor on the prior inside the search (mix with a little uniform); softer policy targets (visit-count temperature above 1 early on); a [Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) root (the call is in `run_search`, commented out), which keeps improving the policy with few simulations.
- Because the outcome is close to chance per run, judging a fix needs several seeds per variant (about 8–10 to tell a failure rate of 1 in 2 from 1 in 10).
- The diagnostic scripts (`diag_split.py`, `run_ckpt.py`) are not in the repository.

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
