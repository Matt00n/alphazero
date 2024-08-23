# Development Progress / change log

[x] Compatibility with Jax environment
[x] Port to Gymnax (convert Gymnax API to Brax (done), modify gymnax to return truncated flags (done), env returns only jax arrays (done))
[x] Ensure that scores are maintained 
[x] dissolve all dependency issues in subfiles
[ ] visualizer
[x] Blueprint of all Muzero elements
[x] Adapt policy learning
[x] Replay Buffers from Brax (logic for prefilling buffer before training)
[x] keep options for both replay & rollout buffer --> test to see what works better in target use case (Temp solution: Comment out sampling from buffer in training_step)
[x] Auto reset not wanted in MCTS rollouts
[x] Fix handling of truncation
[x] Option for GAE to construct value network targets (difficulty: reanalyze --> needs restructuring of buffer to store episodes)
[ ] look into running jax on (M1) GPU: Gymnax, Jax docs, Jax metal
[ ] clean up and organize code / folders
[x] Test with Gymnax environments
[ ] option for different MCTS versions from MCTX
[ ] pass in mcts policy as argument so that we can partial it before
[x] make eval deterministic (as option) 

## Adapt implementation details from further papers: EfficientZero, Reanalyze, Sampled Muzero, Gumble Muzero, AlphaTensor, AlphaDev, DreamerV3

[x] Prioritized replay buffer (see https://github.com/werner-duvaud/muzero-general/tree/master and https://github.com/YeWR/EfficientZero/blob/main/core/replay_buffer.py)
[x] Categorical value function (QR from AlphaTensor instead of MuZeros version)
[x] n-step value targets
[ ] option for learning dynamics model
# NOTE: Muzero samples trajectories during training: unrolls model in parallel to true trajectory and then computes loss for every step. We do not do this since we do not train a model anyway --> apply policy / value to true observation instead of its representation.
[ ] MUZERO REANALYZE: option for reanalyzing data in buffer
[ ] MUZERO REANALYZE: target network (for value), which is updated infrequently
[ ] MUZERO REANALYZE: reanalyze highest reward states (exploit rare events)
[x] SAMPLED MUZERO: sampling of actions / main algorithm
[ ] SAMPLED MUZERO: support for continuous action spaces
[x] ALPHATENSOR: quantile regression distributional loss for value function
[ ] ALPHATENSOR: option for implicit quantile networks (IQN) distributional loss for value function
[ ] ALPHATENSOR: sub-tree persistence --> subtree of selected action is reused in next search
[ ] ALPHATENSOR: bootstrapping during search not with mean value but risk-seeking value (average of quantiles above 75%) (since deterministic env and only interested in best action) (ALSO SEE https://github.com/bwfbowen/muax/tree/main/muax and https://github.com/werner-duvaud/muzero-general/blob/master/models.py for example)



