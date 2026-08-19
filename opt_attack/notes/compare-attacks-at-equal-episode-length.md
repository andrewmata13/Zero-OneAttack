---
name: compare-attacks-at-equal-episode-length
description: "Zero-One attack rewards are only comparable at equal maxlen; budget is per-segment, not per-trajectory"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d92ad1e9-cda4-4038-b191-a04985b9a7d8
  modified: 2026-08-17T00:53:13.096Z
---

Never compare Zero-One attack rewards across different `maxlen` values. Reward accumulates
over the episode, so a short run looks far weaker than it is. Measured on Half Cheetah PPO:
block2+shift at budget 200 reached -54.27 over 200 steps (-0.271 per step) against the
archived -1336.5 over 1000 steps (-1.34 per step) — a 5x gap in rate that says nothing about
the configurations by itself.

Also note **`attack_budget` is per segment, not per trajectory**. Total optimizer calls are
`budget * (maxlen / time_horizon)`, so at maxlen=1000 with H=20 a "budget 200" run performs
50 searches (10,000 evaluations), five times what the same budget buys at maxlen=200.

**Why:** Andrew caught an extrapolation where short-episode sweep results were carried over
to a claim about matching the archived full-length numbers. The A/B comparisons within a
sweep were valid (matched maxlen); the cross-length claim was not.

**How to apply:** Run validation at the reference maxlen=1000 before claiming any budget
reduction, and quote per-step damage rate when episode lengths differ at all. See
[[attack-quality-acceptance-bar]].
