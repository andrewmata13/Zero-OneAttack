---
name: zero-one-ideas-ledger
description: What has been tried on the Zero-One attack and what the measurements said — kept so falsified ideas are not re-proposed
metadata: 
  node_type: memory
  type: project
  originSessionId: d92ad1e9-cda4-4038-b191-a04985b9a7d8
  modified: 2026-08-17T05:17:30.583Z
---

**BEST KNOWN CONFIG (strength objective MET):** `pgd_mode="legacy_norand"`, per-step
(basis=None), no warm start, budget 1500 -> **-1563.9 ± 131.6 at maxlen=1000**, beats
the archived -1336.5 and clears the -1000 bar; all 10 trajectories below -1400. This is
the archive's CrossEntropy loss + deterministic PGD init — see [[pgd-loss-drives-saturation]].

Measured on Half Cheetah PPO. **Kept** (all verified at `maxlen=1000` unless noted):

- **block basis** (`basis=2`): +5.5 damage points, -307.8 vs +158.2 for per-step at budget 200
- **MPC shift warm start** (`warm_start="shift"`): strong at maxlen=200, +0.5 points at 1000;
  only helps in combination with the basis, not with the per-step parameterization
- **deterministic PGD init** (`rand_init=False`) only: worth +35. The rest of the
  "correctness fix" was harmful — see [[pgd-loss-drives-saturation]]
- **trajectory parallelism**: 2.5x on a 4-physical-core box

**Falsified** — do not re-propose without new evidence:

- **early abandonment**: weaker at every margin (-42.6 baseline vs -27.0/-35.1/-38.3), <=22%
  faster. Prefix reward does not predict final reward; the damaging behaviours pay off late
- **learned student warm start**: no gain over an untrained student. Target actions are
  multimodal, so MSE regression collapses to a useless mean
- **terminal value function**: -172 reward, 20x variance. V's RMSE (69-143) exceeded the
  objective's real signal (~10-50), recreating the noisy-objective bug
- **SA-RL prior**: loses to the shift prior, which seeds from an already-optimized plan
- **reachable action box as a defence metric**: falsified. Within-env it goes the wrong way in
  3 of 4 envs; Walker ATLA has 7.9% action authority and still loses 91% of return
- **one-shot learned-dynamics hybrid**: loses to racos at matched simulator cost — see
  [[search-regime-depends-on-dimension]]
- **MSE/targeted PGD ("correctness fix")**: 5x weaker than the archive's CrossEntropy —
  see [[pgd-loss-drives-saturation]]
- **corner/bang-bang target space**: wins 13% at budget 1500, loses 29% at budget 400;
  discrete variables cannot refine locally. Does not give cheaper search
- **basis / block parameterization**: saturates at 88% of archived damage regardless of
  budget (200 -> 3000 moves reward 11 points). Good at low budget, hard ceiling

**Speed status:** strength solved; speed unsolved. Batched-random search FAILED the
budget-1500 linchpin (random 47% of SRACOS). A batchable adaptive search (CEM/CMA-ES)
is the only surviving path to GPU batching and is UNTESTED against SRACOS at budget 1500.
For retraining, the standard fallback is cheap-weak attack in-loop + full -1563 attack
for evaluation, which needs no port.

**Untested:** batched CEM vs SRACOS at budget 1500 (the corrected linchpin); nb_iter=1 (~33% faster, closed form is exact for one step); CEM with model
scoring inside the loop; batched GPU search. Front-loaded budget was proposed twice and
Andrew rejected it from hands-on experience — do not re-propose without asking why.

**Why:** Six structural ideas proposed, five falsified by measurement; the survivors were
unglamorous. Budget is the dominant lever (200 -> 1500 buys +18 damage points; all structure
so far has bought ~6).

**How to apply:** Check this ledger before proposing improvements, and test cheaply before
building. See [[attack-quality-acceptance-bar]].
