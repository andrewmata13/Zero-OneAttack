# GPU-box setup & gates for the JAX/MJX Zero-One port

## Install (i9 + RTX 4080, CUDA 12)
```bash
pip install "jax[cuda12]" mujoco        # mujoco ships MJX
python -c "import jax; print(jax.devices())"      # expect [CudaDevice(0)]
python -c "from mujoco import mjx; print('mjx ok')"
```
Keep the existing torch 2.4/cu121 env for the CPU oracle; JAX only needs the policy
*weights*, not torch.

## Reference numbers to reproduce (Cheetah PPO, maxlen 1000, budget 1500)
| method | reward |
|---|---|
| Zhang SA-RL | -652.9 |
| archived Zero-One | -1336.5 |
| SRACOS legacy_norand | -1563.9 |
| **CEM legacy_norand (target)** | **-1922.3** |
| random | -732.0 |

## The verified contract
`jax_port/policy_ref.py` (pure numpy, runs anywhere) matches the torch policy to 3e-7
and advertorch legacy PGD to 5e-8. `jax_port/attack_mjx.py` mirrors it in jax.numpy.
Run `python jax_port/policy_ref.py` from opt_attack/ to re-verify after any change.

## Config that produced -1922 (bake these in)
- pgd_mode = legacy_norand: CE soft-target loss, UNTARGETED (ascend), rand_init=False
- nb_iter = 1 (== 2 in strength, verified) ; eps_iter = eps = 0.15
- per-step parameterization: dim = 6*20 = 120 (NOT the block basis — 88% ceiling)
- CEM: pop 100, gen 15, elite 15%, init std 0.25, std floor 0.02
- horizon 20, maxlen 1000, frame_skip 5, dt 0.05, reward dx/dt - 0.1*||a||^2
- obs = concat(qpos[1:], qvel) (17d) -> frozen ZFilter (mean/std, clip 10)

## Gates, in order
1. **MJX == C-engine.** Clean rollout of the victim in MJX vs mujoco 3.1.5 C engine;
   behavior/reward must match (~1%). (2.1->3.x behavior transfer already PASSED: 1.01.)
2. **Policy port.** Load weights into the JAX MLP; assert action(obs) == torch to 1e-5.
3. **Strength.** Batched CEM in MJX reproduces ~-1922 on one trajectory (vs the CPU
   oracle). This is the correctness gate.
4. **Attack transfer.** Execute the MJX-found committed plan in the mujoco_py oracle;
   confirm it still degrades the victim to comparable strength. If it doesn't (chaos),
   switch to MJX-search + mujoco_py-commit per segment.
5. **Speed.** Benchmark s/trajectory; target 5-30 s (vs ~18 min CPU-parallel).
6. **Retraining.** Batch across episodes; wire the RL loop on the i9.

## TODOs in attack_mjx.py
- Wire `mjx.step` batching (vmap over the population/episode leading dim).
- Replicate `state0` to the population dim.
- Build the receding-horizon commit loop (MJX or oracle per gate 4).
