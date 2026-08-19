---
name: adv-traj-replay-divergence
description: Replaying saved Adv_Traj trajectories does not reproduce generation-time rewards on Half Cheetah and Ant
metadata: 
  node_type: memory
  type: project
  originSessionId: d92ad1e9-cda4-4038-b191-a04985b9a7d8
  modified: 2026-08-16T23:02:46.030Z
---

`OptimizationAttack.avg_reward()` (and `check_trajectory`) replay saved adversarial observations
through a fresh environment. This reproduces the stored reward **exactly** for Hopper and Walker,
whose episodes terminate early, but **not** for Half Cheetah or Ant, which run the full 1000 steps.

Measured on Cheetah PPO trajectory 0: the perturbation stays inside the eps ball for ~78 steps,
then chaotic divergence takes over — by step 100 the applied perturbation is 0.33 in normalized
units against a 0.015 budget, so it is no longer a valid bounded attack. Divergence is not
systematically weaker: Cheetah LSTM replays at -270 vs 3099 stored, Ant LSTM at 710 vs 1177.

The saved pickles appear to predate the released `generate()` — the dead `input_state` variable in
that method hints the confirmation loop once worked differently.

**Why:** Replay-based numbers on the 1000-step environments are unreliable in either direction and
should not be used for the paper comparison.

**How to apply:** Cite `traj.reward` for Cheetah/Ant; `avg_reward()` is trustworthy only for Hopper
and Walker. See [[attack-budget-and-adv-traj]].
