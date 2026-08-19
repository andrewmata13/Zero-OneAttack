---
name: attack-budget-and-adv-traj
description: The Adv_Traj pickles are the high-budget paper results; attack_budget is the dominant quality knob and is not recorded anywhere
metadata: 
  node_type: memory
  type: project
  originSessionId: d92ad1e9-cda4-4038-b191-a04985b9a7d8
  modified: 2026-08-16T23:02:37.286Z
---

In Zero-One Attack, `attack_budget` (the zoopt `Parameter(budget=...)` simulation count) is the
dominant knob for attack strength, and it is **not stored** in the saved trajectories —
`AdversarialTrajectory` keeps only start_state, adv_states, adv_actions, timing, reward.

The pickles in `opt_attack/sim_suite/Adv_Traj/` are the high-budget runs behind the ICCPS 2024
paper. Their `traj.reward` field is the number to cite. `custom_sim.py` ships
`attack_budget = 10`, roughly two orders of magnitude below what those traces used (the Cheetah
state-attack decision vector alone is dim 120 = 6 actions x H=20).

Measured effect at eps=0.15 on Half Cheetah PPO: budget-10 replay reaches only -106 average
reward, versus -1337 in the stored high-budget trace.

**Why:** Andrew flagged that low-budget reproductions badly understate the method, and asked that
the saved traces be treated as the reference results.

**How to apply:** When quoting Zero-One performance, read `traj.reward` from the Adv_Traj pickles
rather than re-running with default parameters, and state the budget whenever reporting a fresh
run. See [[adv-traj-replay-divergence]].
