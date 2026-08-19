'''
Entry point for a Zero-One attack run.

Must be run from the opt_attack directory, since the victim checkpoints are
loaded by relative path:

    python3.8 run_attack.py --env Cheetah --net PPO --budget 200 --traj 10

Defaults are the BEST measured config on Half Cheetah PPO (budget 1500, maxlen
1000): per-step parameterization + CEM search + legacy_norand PGD -> reward -1922
(beats archived -1336.5). legacy_norand = the archive's CrossEntropy/untargeted
loss + deterministic init; the "faithful" MSE/targeted variant is 9x WEAKER. The
block basis caps at 88% of archived damage regardless of budget - do not use it to
reach the -1000 bar. See CLAUDE.md for the full session findings.
'''

import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from optimization_attack import OptimizationAttack

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Hopper.Hopper import Hopper
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Ant.AntEnv import Ant

ENVS = {"Cheetah": HalfCheetah, "Hopper": Hopper, "Walker": Walker2D, "Ant": Ant}

# Reference numbers measured on 10 trajectories: clean, Zhang's learned adversary,
# and the archived Zero-One traces (budget 1500). Lower is a stronger attack.
REFERENCE = {
    ("Cheetah", "PPO"):  (7163.9,  -652.9, -1336.5),
    ("Cheetah", "ATLA"): (5647.7,  3087.5, -1707.6),
    ("Cheetah", "LSTM"): (6623.6,  4985.2,  3098.5),
    ("Hopper",  "PPO"):  (3043.2,   638.6,   762.7),
    ("Hopper",  "ATLA"): (2400.6,  1019.5,   684.0),
    ("Hopper",  "LSTM"): (3401.9,  1466.0,  1212.3),
    ("Walker",  "PPO"):  (4634.8,  1052.5,   743.2),
    ("Walker",  "ATLA"): (3287.9,  2087.3,   293.1),
    ("Walker",  "LSTM"): (4024.0,  3689.0,  1246.3),
    ("Ant",     "PPO"):  (5982.7,   167.2,  -407.6),
    ("Ant",     "ATLA"): (4634.9,  -201.8,  -406.3),
    ("Ant",     "LSTM"): (5331.5,  3566.0,  1177.1),
}


# Absolute strength targets a run must reach to be acceptable. Set where an
# explicit bar has been agreed; percentage-of-archived is reported otherwise.
# Note the percentage metric is lenient near the top: on Cheetah PPO the Zhang
# baseline already sits at 92% of archived damage, so anything under ~92% loses
# to the baseline it is supposed to beat.
TARGET = {
    ("Cheetah", "PPO"): -1000.0,
}


def horizon_for(params, net):
    return params["time_horizon_%s" % net]


def main():
    p = argparse.ArgumentParser(description="Run a Zero-One attack")
    p.add_argument("--env", default="Cheetah", choices=sorted(ENVS))
    p.add_argument("--net", default="PPO", choices=["PPO", "ATLA", "LSTM"])
    p.add_argument("--attack", default="state", choices=["state", "time", "state+time"])
    p.add_argument("--budget", type=int, default=200, help="zeroth order simulations per segment")
    p.add_argument("--traj", type=int, default=10, help="number of trajectories")
    p.add_argument("--maxlen", type=int, default=None, help="episode length (default: env setting)")
    p.add_argument("--horizon", type=int, default=None, help="look-ahead (default: env setting)")
    p.add_argument("--basis", type=int, default=0, help="0 = per-step (best); k = block basis (88%% ceiling, low-budget only)")
    p.add_argument("--basis-type", default="block", choices=["block", "dct"])
    p.add_argument("--abandon", type=float, default=None, help="early abandonment margin; omit to disable")
    p.add_argument("--warm-start", default=None, choices=[None, "shift"])
    p.add_argument("--legacy", action="store_true", help="reproduce the archived PGD settings")
    p.add_argument("--target", type=float, default=None,
                   help="reward the run must reach to pass (default: the agreed bar for this config)")
    p.add_argument("--optimizer", default="cem", choices=["racos", "random", "cem", "hybrid"],
                   help="zeroth-order search (cem is strongest+batchable: -1922 on Cheetah PPO)")
    p.add_argument("--pgd-mode", dest="pgd_mode", default="legacy_norand",
                   choices=["legacy", "legacy_norand", "fixed", "fixed_rand"],
                   help="inner PGD variant; legacy_norand (CE + deterministic init) is strongest")
    p.add_argument("--cem-pop", dest="cem_pop", type=int, default=100)
    p.add_argument("--cem-elite", dest="cem_elite", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel trajectory workers (0 = one per core)")
    p.add_argument("--save", default=None, help="path to pickle the trajectories to")
    args = p.parse_args()

    if not os.path.isdir("envs"):
        sys.exit("run this from the opt_attack directory (victim checkpoints are relative paths)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctor = ENVS[args.env]
    params = ctor(args.net).params
    horizon = args.horizon or horizon_for(params, args.net)
    maxlen = args.maxlen or params["maxlen"]
    basis = args.basis if args.basis and args.basis > 0 else None

    print("%s %s  attack=%s  horizon=%d  maxlen=%d  eps=%g" %
          (args.env, args.net, args.attack, horizon, maxlen, params["eps"]))
    dim = (len(ctor(args.net).actionBounds) * (basis if basis else horizon))
    pgd_label = "legacy" if args.legacy else (args.pgd_mode or "fixed")
    print("  budget=%d  basis=%s  dim=%d  evals/dim=%.1f  optimizer=%s  pgd=%s  warm_start=%s" %
          (args.budget, basis or "per-step", dim, args.budget / float(dim),
           args.optimizer, pgd_label, args.warm_start))

    attack = OptimizationAttack(
        environment=ctor,
        start_state=params["start_state"],
        eps=params["eps"],
        num_iter=params["num_iter"],
        step_size=params["step_size"],
        time_horizon=horizon,
        maxlen=maxlen,
        net_type=args.net,
        attack_type=args.attack,
        attack_budget=args.budget,
        basis=basis,
        basis_type=args.basis_type,
        abandon_margin=args.abandon,
        warm_start=args.warm_start,
        legacy_pgd=args.legacy,
        pgd_mode=args.pgd_mode,
        optimizer=args.optimizer,
        cem_pop=args.cem_pop,
        cem_elite=args.cem_elite,
    )

    t0 = time.time()
    if args.workers != 1 and args.traj > 1:
        from parallel_attack import run_parallel
        import multiprocessing
        workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()
        print("  running %d trajectories across %d workers" % (args.traj, workers))
        trajs, evaluated, abandoned = run_parallel(
            env=args.env, net=args.net, attack=args.attack, horizon=horizon,
            maxlen=maxlen, budget=args.budget, num_traj=args.traj, basis=basis,
            basis_type=args.basis_type, abandon=args.abandon,
            warm_start=args.warm_start, legacy=args.legacy, seed=args.seed,
            optimizer=args.optimizer, pgd_mode=args.pgd_mode,
            cem_pop=args.cem_pop, cem_elite=args.cem_elite,
            workers=workers)
        attack.adv_trajs = trajs
        attack.evaluated, attack.abandoned = evaluated, abandoned
    else:
        attack.generate(num_traj=args.traj)
    elapsed = time.time() - t0

    rewards = [t.reward for t in attack.adv_trajs]
    print()
    print("=" * 62)
    print("attacked reward   %9.1f +- %.1f  over %d trajectories" %
          (np.mean(rewards), np.std(rewards), len(rewards)))

    ref = REFERENCE.get((args.env, args.net))
    comparable = (maxlen == params["maxlen"] and args.attack == "state")
    if ref is not None and not comparable:
        print("(reference numbers are for attack=state at maxlen=%d, not comparable to this run)"
              % params["maxlen"])
    elif ref is not None:
        clean, zhang, archived = ref
        got = np.mean(rewards)
        print("clean             %9.1f" % clean)
        print("Zhang (SA-RL)     %9.1f   %s" %
              (zhang, "BEATEN" if got < zhang else "not beaten"))
        print("archived Zero-One %9.1f" % archived)

        ###################################################################
        # Acceptance bar: a cheaper configuration is only useful if it    #
        # lands within 10-20% of the archived attack's damage. Damage is  #
        # measured from clean, so that a sign change in reward does not   #
        # distort the ratio.                                              #
        ###################################################################
        damage, full = clean - got, clean - archived
        pct = 100.0 * damage / full if full else float("nan")
        print("damage vs archived %8.1f%%" % pct)

        bar = args.target if args.target is not None else TARGET.get((args.env, args.net))
        if bar is not None:
            print("target            %9.1f   %s" %
                  (bar, "PASS" if got <= bar else "FAIL (short by %.1f)" % (got - bar)))

    if attack.evaluated:
        print("candidates        %9d evaluated, %d pruned (%.1f%%)" %
              (attack.evaluated, attack.abandoned,
               100.0 * attack.abandoned / attack.evaluated))
    print("elapsed           %9.1fs  (%.1fs per trajectory)" % (elapsed, elapsed / max(len(rewards), 1)))
    print("=" * 62)

    if args.save:
        attack.save_attack(args.save)
        print("saved to %s" % args.save)


if __name__ == "__main__":
    main()
