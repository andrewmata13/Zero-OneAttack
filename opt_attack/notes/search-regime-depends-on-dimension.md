---
name: search-regime-depends-on-dimension
description: "Random search ≈ SRACOS" only holds at the full per-step parameterization; with the block basis the adaptive search genuinely works
metadata:
  type: feedback
---

Measured on Half Cheetah PPO: at the original per-step parameterization (dim 120, budget
below or near the dimension) uniform random sampling matches SRACOS, so the zeroth order
search contributes almost nothing. **This does not carry over to the block basis.** At
`basis=2` the dimension is 12, budget 200 gives ~17 evaluations per dimension, and the
adaptive search clearly beats random sampling.

This invalidated the learned-dynamics hybrid. Its ranking checks compared "model shortlist"
against "random selection from a random pool" and won in 8 of 8 cells — but the real baseline
is an *adaptive* search that finds candidates better than anything in a random pool. At
matched MuJoCo cost (250 rollouts) racos scored -27.64 and the one-shot hybrid only -6.50.

**Why:** A measurement taken in one regime (dim 120) was carried into another (dim 12) where
it no longer holds. Same class of error as comparing rewards across different maxlen.

**How to apply:** Before using random sampling as a proxy for the search, check evaluations
per dimension for the *current* parameterization. Any model-guided method must be benchmarked
against adaptive search at matched simulator cost, never against random selection. See
[[compare-attacks-at-equal-episode-length]] and [[zero-one-ideas-ledger]].


**Budget-dependent too (measured budget 1500, dim 120, maxlen 1000, legacy_norand):**
SRACOS -1563.9 vs random -732.0 — SRACOS is 2x stronger. So "random ≈ SRACOS" holds
only at LOW budget (200-600); at the budget that reaches full strength, the adaptive
search matters a great deal. This killed the "replace SRACOS with batched best-of-N
random" speed plan. A batchable search must be ADAPTIVE (CEM/CMA-ES), not random.
