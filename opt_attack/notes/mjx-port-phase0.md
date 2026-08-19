---
name: mjx-port-phase0
description: Phase 0 findings for the JAX/MJX GPU port — victim behavior transfers to modern MuJoCo, so victims are kept
metadata:
  type: project
---

Plan: port the CEM attack to JAX + MJX on the i9+4080 for ~100x speedup (evaluation
minutes not hours; retraining feasible). CEM at budget 1500 scored **-1922** (beats
SRACOS -1563.9 AND is batchable) — it is the search to build the port around.

**Phase 0 physics gate — PASSED (measured on the laptop, no GPU here):**
- Victim env is standard HalfCheetah, mujoco_py 2.1.2, nq=nv=9 nu=6, timestep 0.01,
  frame_skip 5. MJCF at gym/envs/mujoco/assets/half_cheetah.xml.
- mujoco 3.1.5 is installed; jax/MJX are NOT (install on the i9 box).
- MuJoCo 2.1 vs 3.1 STATE trajectories diverge (8e-2 qpos over 20 steps) despite
  identical solver opts — intrinsic engine difference through HalfCheetah chaos.
- BUT the victim policy driven through mujoco 3.1.5 gives clean reward **7360 vs
  7307** in its native engine (ratio 1.01). **Behavior transfers; victims are kept,
  not retrained.** The divergence is trajectory noise a feedback controller corrects.

**Still to confirm (Phase 1, on the i9+4080):** MJX (JAX reimpl) matches the mujoco
3.x C engine (small gap, designed to); ATTACK transfer (an attack searched in the
new stack degrades the victim to comparable strength). Port the 64x64 policy to JAX
(weights copy, verify action match 1e-5); ZFilter is frozen-per-segment so trivial.

**How to apply:** the port is viable with existing victims. Build the batched CEM
loop in JAX/MJX; keep the CPU pipeline as the reference oracle producing -1922 and
the trajectory-of-record. See [[zero-one-ideas-ledger]] and [[pgd-loss-drives-saturation]].
