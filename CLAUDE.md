# Zero-One Attack — project state & continuation guide

Adversarial attack on RL locomotion controllers (Zhang et al. victims: HalfCheetah,
Hopper, Walker2D, Ant; PPO/ATLA/LSTM). Published as ICCPS 2024. This guide is the
entry point for continuing the **speed/retraining** work, especially the JAX/MJX
GPU port. Detailed rationale behind each decision is in opt_attack/notes/ (start at
notes/INDEX.md) — these travel with the repo.

## The method in one paragraph
Receding-horizon attack. An OUTER zeroth-order search proposes a *target action
sequence* over a 20-step lookahead (dim 120 = 6 actions x 20 steps); an INNER PGD
solve turns each target into a bounded observation perturbation (eps=0.15 in
normalized obs space) that makes the victim emit ~that action. Commit the best
segment, advance, re-search. Budget = evaluations per segment; total cost =
budget x maxlen candidate-steps.

## KEY FINDINGS THIS SESSION (these overturn the original code)
1. **The original PGD was "wrong" and that is WHY it works.** advertorch defaults:
   CrossEntropy loss on the continuous action, UNTARGETED (ascends AWAY from target),
   random init. "Fixing" it to faithful MSE/targeted cost a **9x** strength loss,
   because CE saturates actions to the rails (35.8% vs 20.2%) and bang-bang actions
   break the gait. Faithful targeting aims at moderate actions (mean |target|=0.5).
   => use `pgd_mode="legacy_norand"` (CE + deterministic init). Deterministic init
   (rand_init=False) is the ONE good part of the "fix": +35 reward.
2. **CEM beats SRACOS and is batchable.** At budget 1500: CEM **-1922.3** vs SRACOS
   -1563.9 vs random -732.0. Budget curve (CEM, pop 100): b1500 -1922, b800 -1367.8
   (still beats archived -1336.5), b500 -744.9 (below archived). CEM reaches archived
   strength at ~HALF budget; the floor is ~700-800, below which it degrades fast.
   Per-generation population is independent -> GPU batchable. Build the port on CEM.
3. **nb_iter=1 == nb_iter=2** (-1557 vs -1564). The closed-form single sign step
   delta=eps*sign(grad) is valid; curvature is irrelevant.
4. **Theory:** the 120-dim target is a SEED, not a plan (corr(target,realized)~0.5).
   CE-PGD maps it chaotically onto the SATURATED BOUNDARY of the reachable action set
   (measured 26-46% of the action box). Search = finding the seed whose bang-bang
   sequence tips the plant into the reverse-gait basin. Dimension = temporal
   resolution of the bang-bang sequence (why block basis caps at 88%). Velocity plot:
   the attack captures the reverse regime within ~50 of 1000 steps and holds it flat
   (no mid-episode transition) - the outcome is decided in the opening segments. Two
   theory predictions confirmed: CEM>SRACOS (big populations search seeds better) and
   nb_iter=1==nb_iter=2 (curvature irrelevant).
5. **Falsified ideas (do not retry):** block/basis parameterization (88% ceiling),
   early abandonment, learned student, terminal value function, SA-RL prior,
   reachable-box defence metric, one-shot learned-dynamics hybrid, corner targets,
   front-loaded budget (rejected from experience). Full numbers: notes/zero-one-ideas-ledger.md.

## Best config + reference (Cheetah PPO, maxlen 1000, budget 1500)
Zhang -652.9 | archived -1336.5 | SRACOS legacy_norand -1563.9 | **CEM -1922.3** |
random -732.0. Target bar: reward <= -1000.

Reproduce the best result (this is now the DEFAULT config of run_attack.py):
```bash
cd opt_attack
python3.8 run_attack.py --env Cheetah --net PPO --budget 1500 --traj 10 --workers 8
# defaults: --basis 0 (per-step) --optimizer cem --pgd-mode legacy_norand  -> ~ -1922
# archive repro: add --legacy --optimizer racos  (or load the saved pickle below)
```
`run_attack.py` self-scores vs the -1000 bar. `parallel_attack.run_parallel(...)` takes the
same knobs as kwargs (optimizer, pgd_mode, cem_pop, cem_elite, num_iter_override, basis,
warm_start) for scripting.

## Saved evidence (durable, in opt_attack/sim_suite/Adv_Traj/)
Load with pickle to verify without re-running (each is 10 trajectories, maxlen 1000, b1500):
- `Cheetah_PPO_cem_norand_b1500`      -1922.3  (best; CEM + legacy_norand)
- `Cheetah_PPO_legacy_norand_b1500`   -1563.9  (SRACOS + legacy_norand)
- `Cheetah_PPO_random_norand_b1500`    -732.0  (random control)
- `Cheetah_PPO_nbiter1_norand_b1500`  -1557.4  (nb_iter=1, == nb_iter=2)
- `Cheetah_PPO_cem_b800`               -1367.8  (CEM at HALF budget still beats archived)
- `Cheetah_PPO_cem_b500`                -744.9  (below archived; budget floor is ~700-800)
- `Cheetah_PPO_block2shift_b{400,800,1500,3000}`  the basis 88%-ceiling curve
(pickle protocol needs `sys.modules['__main__'].AdversarialTrajectory = util.AdversarialTrajectory`)

## Environments
- **Oracle (this + CPU work):** python3.8, torch 2.4.1+cu121, gym 0.22, mujoco_py
  2.1.2.14, zoopt, advertorch. Runs the victims and the trajectory of record.
- **Port (GPU box):** jax[cuda12] + mujoco 3.x (ships MJX). mujoco 3.1.5 already
  present on the CPU box; jax is NOT. See jax_port/SETUP.md.

## The port — jax_port/
- `policy_ref.py`  numpy reference for policy + legacy_norand PGD. VERIFIED vs torch
  (3e-7) and advertorch (5e-8). Runs anywhere: `python3.8 jax_port/policy_ref.py`.
- `attack_mjx.py`  batched CEM+MJX skeleton mirroring the reference. Two TODOs (MJX
  step batching). Everything else complete.
- `SETUP.md`  install, the 6 gates in order, config constants, reference numbers.
Phase 0 physics gate PASSED: victim behavior transfers 2.1->3.1 at ratio 1.01, so
victims are KEPT (not retrained). Remaining: MJX==C-engine, policy port, strength
reproduction, attack transfer, speed, retraining.

## Projected speed after CEM + JAX/MJX on i9+4080
Per-trajectory ~5-30s (vs ~18 min CPU-parallel today); 10-traj eval ~1-5 min;
retraining 2000 episodes ~1-3 h. Serial floor per trajectory = maxlen*generations =
1000*15 = 15k batched steps. GPU collapses the population dimension; i9 orchestrates.

## Gotchas
- `envs/Hopper/Hopper.py` had Ant's policy dims (111,8); FIXED to (11,3). All victims load.
- Save paths: pickles live under `opt_attack/sim_suite/Adv_Traj/`, NOT `opt_attack/Adv_Traj/`.
- Compare rewards ONLY at equal maxlen; budget is PER-SEGMENT (total = budget*maxlen/H).
- Cheetah/Ant saved traces do NOT replay faithfully (chaos); cite generation-time reward.
- Detached-job waiters: don't `pgrep -f <script>.py` from the waiter's own command
  (self-match, never exits) — poll the LOG for a verdict line instead.
- Reward = dx/dt - 0.1*||a||^2, dt=0.05. obs = concat(qpos[1:], qvel), 17d, ZFilter clip 10.
- Target bar is reward <= -1000, NOT a %-of-archived band: Zhang already sits at 92% of
  archived damage, so an 80-90% band would pass attacks that lose to the baseline.
- Already-applied code fixes this session: Hopper dims; dead u0 removed; L-inf assert
  corrected; Walker early-stop wired to the confirmation rollout. These are in the tree.

## File map (opt_attack/)
- optimization_attack.py  the attack. optimizer racos|random|cem|hybrid;
  pgd_mode legacy|legacy_norand|fixed|fixed_rand; basis None=per-step; warm_start shift|sarl.
- parallel_attack.py  trajectory-level multiprocessing (run_parallel).
- run_attack.py  CLI + PASS/FAIL vs target bar.
- hybrid.py value.py student.py  falsified approaches (kept, unused).
- envs/*/  victim wrappers; sim_suite/Adv_Traj/  saved traces.
- notes/  session findings & rationale (INDEX.md + 8 topic notes), self-contained.
