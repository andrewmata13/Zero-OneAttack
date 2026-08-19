---
name: pgd-loss-drives-saturation
description: The archive's "wrong" CrossEntropy PGD loss works because it saturates actions; the faithful targeted attack is 5x weaker
metadata:
  type: project
---

The PGD inner solve in `optimization_attack.py` used advertorch defaults:
`CrossEntropyLoss` on the continuous action vector, `targeted=False` (gradient
ascent AWAY from the target), `rand_init=True`. That looks like a plain bug — the
paper says PGD drives the policy toward the target action.

**"Fixing" it cost a factor of nine.** Isolated 2x2 on Cheetah PPO, maxlen=100,
budget 1500, per-step, no warm start:

    legacy         -73.69   CE/untargeted | random init
    legacy_norand -120.85   CE/untargeted | deterministic   <- BEST KNOWN
    fixed_rand      -0.85   MSE/targeted  | random init
    fixed          -24.57   MSE/targeted  | deterministic

    main effect rand_init:      +35.44  (randomisation HURTS)
    main effect loss/targeting: -84.56  (CrossEntropy HELPS)

**Mechanism (measured, not inferred):** targets are drawn uniformly from [-1,1]^6,
mean magnitude 0.5, so a *faithful* targeted attack spends most of its effort
steering the victim toward moderate actions. CrossEntropy/untargeted ignores target
magnitude and pushes outputs to the rails regardless. Measured saturated-action rate
35.8% (CE) vs 20.2% (MSE), reachable action spread 0.66 vs 0.55. Bang-bang actions
are what break a locomotion gait.

Restricting the search to corner targets {-1,+1} does NOT exploit this usefully:
it wins ~13% at budget 1500 but loses ~29% at budget 400 (discrete search cannot
refine locally), and neither gap is significant at n=10.

**Why:** three changes were bundled as one "correctness fix" and validated at
maxlen=40, where the effect is invisible. maxlen=100 does show it (verified against
maxlen=1000 anchors at 7.4x vs 9.2x).

**How to apply:** default to `pgd_mode="legacy_norand"`. If the semantics need to be
defensible in writing, the honest formulation is "maximize s·pi(x+delta) for a sign
vector s", which states what CE does by accident. See [[zero-one-ideas-ledger]].
