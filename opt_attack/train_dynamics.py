'''
Pre-train a dynamics ensemble for the hybrid search.

Data comes from the distribution the search explores - rollouts driven by random
target action sequences through the PGD inner solve - sampled from states spread
along a clean trajectory.

    python3.8 train_dynamics.py --env Cheetah --net PPO --rollouts 5000 \
        --out dynamics_cheetah_ppo.pt
'''

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from torch import nn
from advertorch.attacks import PGDAttack

from hybrid import DynamicsEnsemble
from util import AdvertorchAdapter

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Hopper.Hopper import Hopper
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Ant.AntEnv import Ant

ENVS = {"Cheetah": HalfCheetah, "Hopper": Hopper, "Walker": Walker2D, "Ant": Ant}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Cheetah", choices=sorted(ENVS))
    p.add_argument("--net", default="PPO", choices=["PPO", "ATLA"])
    p.add_argument("--rollouts", type=int, default=5000)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--basis", type=int, default=2)
    p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not os.path.isdir("envs"):
        sys.exit("run this from the opt_attack directory")

    out = args.out or "dynamics_%s_%s.pt" % (args.env.lower(), args.net.lower())
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)

    h = ENVS[args.env](args.net)
    params = h.params
    na = len(h.actionBounds)
    pgd_model = AdvertorchAdapter(ENVS[args.env](args.net).model)
    pgd_model.eval()
    env = h.env
    H = args.horizon

    attack = PGDAttack(pgd_model, loss_fn=nn.MSELoss(reduction="sum"), eps=params["eps"],
                       nb_iter=params["num_iter"], eps_iter=params["step_size"],
                       clip_min=-1000, clip_max=1000, targeted=True, rand_init=False)

    # states spread along a clean trajectory, so the model sees the whole gait
    obs = env.reset(params["start_state"], None)
    states = [np.array(params["start_state"], dtype=np.float64)]
    for _ in range(700):
        st, _, done, _ = env.step(env.predict(obs), change_filter=True)
        obs = st[1]
        states.append(np.array(st[0], dtype=np.float64))
        if done:
            obs = env.reset(params["start_state"], None)

    print("collecting %d rollouts of %d steps" % (args.rollouts, H), flush=True)
    t0 = time.time()
    S, A, D = [], [], []
    for r in range(args.rollouts):
        s0 = states[rng.randint(len(states))]
        coeffs = rng.uniform(-1, 1, args.basis * na).reshape(args.basis, na)
        targets = coeffs[(np.arange(H) * args.basis) // H]

        obs = env.reset(s0, None)
        u = np.array(s0, dtype=np.float64)
        for i in range(H):
            y = torch.tensor(np.array([targets[i]])).float()
            adv = attack.perturb(torch.tensor(obs).float().unsqueeze(0), y=y).detach().numpy()[0]
            a = env.predict(adv)
            st, _, done, _ = env.step(a)
            un = np.array(st[0], dtype=np.float64)
            S.append(u.copy())
            A.append(np.array(a).flatten()[:na])
            D.append(un - u)
            u, obs = un, st[1]
            if done:
                break
        if (r + 1) % max(1, args.rollouts // 10) == 0:
            print("  %d/%d rollouts, %d transitions, %.0fs"
                  % (r + 1, args.rollouts, len(S), time.time() - t0), flush=True)

    S, A, D = np.array(S), np.array(A), np.array(D)
    print("collected %d transitions in %.0fs" % (len(S), time.time() - t0), flush=True)

    ens = DynamicsEnsemble(S.shape[1], A.shape[1], members=args.members, hidden=args.hidden)
    t0 = time.time()
    ens.fit(S, A, D, epochs=args.epochs, verbose=True)
    print("trained in %.0fs" % (time.time() - t0), flush=True)

    # held-out one-step sanity check
    n = len(S) // 10
    with torch.no_grad():
        x = torch.tensor(S[-n:], dtype=torch.float32)[:, 1:]
        a = torch.tensor(A[-n:], dtype=torch.float32)
        y = torch.tensor(D[-n:], dtype=torch.float32)
        pred = torch.stack([m(x, a) for m in ens.nets]).mean(0)
    rel = ((pred - y).abs().mean(0) / (y.abs().mean(0) + 1e-8))
    print("held-out one-step relative error: mean %.3f  worst dim %.3f"
          % (float(rel.mean()), float(rel.max())), flush=True)

    ens.save(out)
    print("saved %s" % out, flush=True)


if __name__ == "__main__":
    main()
