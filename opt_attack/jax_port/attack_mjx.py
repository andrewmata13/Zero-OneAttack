'''
Batched CEM + MJX Zero-One attack  (GPU box: needs jax[cuda12] + mujoco/mjx).

Mirrors jax_port/policy_ref.py (verified against torch to 1e-7). The whole candidate
population steps in lockstep: one batched PGD + policy + MJX step per timestep, so
the population dimension is free on the GPU. This is the fast path; the CPU pipeline
(../optimization_attack.py, optimizer="cem", pgd_mode="legacy_norand") is the oracle
that produces the reference -1922 and the trajectory of record.

STATUS: skeleton. Fill the two TODOs (MJX batched step wiring) and run the gates in
SETUP.md. Everything else (policy, PGD, CEM, reward) is complete and mirrors the
verified numpy reference.
'''
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

# ---- constants for HalfCheetah PPO (see SETUP.md / CLAUDE.md) ----
EPS = 0.15
HORIZON = 20
MAXLEN = 1000
DT = 0.05            # timestep 0.01 * frame_skip 5
FRAME_SKIP = 5
CTRL_COST = 0.1
NB_ITER = 1          # nb_iter=1 == nb_iter=2 in strength (verified); 1 is cheaper
CLIP_OBS = 10.0      # ZFilter clip


# ---------- policy + legacy_norand PGD (mirror of policy_ref.py) ----------
def policy_mean(w, x):
    h = jnp.tanh(x @ w["W0"].T + w["b0"])
    h = jnp.tanh(h @ w["W1"].T + w["b1"])
    return h @ w["Wm"].T + w["bm"]

def _ce_loss(delta, w, x, y):
    logits = policy_mean(w, x + delta)
    logp = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.sum(y * logp)          # summed over batch+dims, like reduction='sum'

def legacy_pgd(w, x, y, eps=EPS, nb_iter=NB_ITER):
    'Batched. x:[N,17] obs, y:[N,6] target actions -> adv obs [N,17].'
    grad_fn = jax.grad(_ce_loss)       # d(sum loss)/d delta == per-row grad, batched
    delta = jnp.zeros_like(x)
    for _ in range(nb_iter):
        g = grad_fn(delta, w, x, y)
        delta = jnp.clip(delta + eps * jnp.sign(g), -eps, eps)
    return jnp.clip(x + delta, -1000.0, 1000.0)

def normalize(u_qvel_obs, mean, std):
    'Frozen ZFilter: (raw_obs - mean)/std, clipped. raw_obs = concat(qpos[1:], qvel).'
    o = (u_qvel_obs - mean) / (std + 1e-8)
    return jnp.clip(o, -CLIP_OBS, CLIP_OBS)


# ---------- batched segment rollout through MJX ----------
def rollout_population(w, mjx_model, state0, targets, mean, std):
    '''Score a whole population over one HORIZON-step segment in MJX.
    targets: [N, HORIZON, 6] proposed target actions.  Returns reward [N].
    state0 : batched MJX data replicated to N (same start for all candidates).

    TODO(GPU): wire the two MJX calls below to the installed mjx API:
      - mjx.step(mjx_model, data) batched over N via jax.vmap
      - read qpos[:,0] (root x) and construct raw_obs = concat(qpos[:,1:], qvel)
    The rest (normalize -> PGD -> policy -> reward) is complete.
    '''
    from mujoco import mjx  # noqa: import here so the file loads without mujoco
    N = targets.shape[0]
    data = state0                      # batched MjxData with leading dim N

    total = jnp.zeros(N)
    step = jax.vmap(mjx.step, in_axes=(None, 0))
    for i in range(HORIZON):
        qpos = data.qpos; qvel = data.qvel
        raw_obs = jnp.concatenate([qpos[:, 1:], qvel], axis=-1)     # [N,17]
        obs = normalize(raw_obs, mean, std)
        adv = legacy_pgd(w, obs, targets[:, i, :])
        act = jnp.clip(policy_mean(w, adv), -1.0, 1.0)              # executed action

        x0 = data.qpos[:, 0]
        data = data.replace(ctrl=act)
        for _ in range(FRAME_SKIP):
            data = step(mjx_model, data)                           # TODO verify API
        total = total + (data.qpos[:, 0] - x0) / DT - CTRL_COST * jnp.sum(act**2, -1)
    return total, data


# ---------- CEM over the seed space, per segment ----------
def cem_segment(w, mjx_model, state0, mean_stats, std_stats, key,
                pop=100, gen=15, elite_frac=0.15, dim=HORIZON * 6):
    'Returns the best target sequence [HORIZON,6] found for this segment.'
    lo, hi = -1.0, 1.0
    cmean = jnp.zeros(dim); cstd = jnp.full(dim, (hi - lo) / 4.0)
    n_elite = max(2, int(round(pop * elite_frac)))
    best_seq, best_r = None, jnp.inf
    for g in range(gen):
        key, sk = jax.random.split(key)
        samples = jnp.clip(cmean + cstd * jax.random.normal(sk, (pop, dim)), lo, hi)
        targets = samples.reshape(pop, HORIZON, 6)
        # replicate state0 to pop; TODO(GPU): batch state0 to leading dim pop
        rewards, _ = rollout_population(w, mjx_model, state0, targets, mean_stats, std_stats)
        order = jnp.argsort(rewards)
        elite = samples[order[:n_elite]]
        cmean = elite.mean(0)
        cstd = jnp.maximum(elite.std(0), 0.02 * (hi - lo))
        if rewards[order[0]] < best_r:
            best_r = rewards[order[0]]; best_seq = targets[order[0]]
    return best_seq, best_r


# receding horizon over MAXLEN/HORIZON segments = the full attack; commit each best
# segment (in MJX, or in the mujoco_py oracle if Phase 1 shows attack transfer needs it)
# is the remaining wiring — see SETUP.md Phase 1.
