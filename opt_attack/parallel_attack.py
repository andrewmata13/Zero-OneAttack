'''
Trajectory-level parallelism.

Attacking N trajectories is N independent problems, so they run in separate
processes. Each worker rebuilds the full seeded starting-state list and attacks
only its own index, so the starting states match a sequential run exactly.

Results are not bit-identical to a sequential run: the zeroth order search draws
from the global RNG, which a sequential run advances across trajectories. Each
worker is seeded deterministically from its index instead, so a parallel run is
reproducible on its own terms.
'''

import multiprocessing as mp
import random
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


def _worker(job):
    'Attack a single trajectory. Must be top level so it can be pickled.'
    from optimization_attack import OptimizationAttack
    from envs.HalfCheetah.CheetahEnv import HalfCheetah
    from envs.Hopper.Hopper import Hopper
    from envs.Walker2D.Walker2DEnv import Walker2D
    from envs.Ant.AntEnv import Ant

    envs = {"Cheetah": HalfCheetah, "Hopper": Hopper, "Walker": Walker2D, "Ant": Ant}

    # One thread per worker: these nets are tiny and oversubscribing the cores
    # costs more than the intra-op parallelism gains
    torch.set_num_threads(1)

    index = job["index"]
    seed = job["seed"] + index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ctor = envs[job["env"]]
    params = ctor(job["net"]).params

    attack = OptimizationAttack(
        environment=ctor,
        start_state=params["start_state"],
        eps=params["eps"],
        num_iter=job.get("num_iter_override") or params["num_iter"],
        step_size=params["step_size"],
        time_horizon=job["horizon"],
        maxlen=job["maxlen"],
        net_type=job["net"],
        attack_type=job["attack"],
        attack_budget=job["budget"],
        basis=job["basis"],
        basis_type=job["basis_type"],
        abandon_margin=job["abandon"],
        warm_start=job["warm_start"],
        legacy_pgd=job["legacy"],
        pgd_mode=job.get("pgd_mode"),
        target_space=job.get("target_space", "continuous"),
        optimizer=job.get("optimizer", "racos"),
        cem_pop=job.get("cem_pop", 100),
        cem_elite=job.get("cem_elite", 0.15),
    )

    attack.generate(num_traj=job["num_traj"], only_index=index)

    return {
        "index": index,
        "traj": attack.adv_trajs[0],
        "evaluated": attack.evaluated,
        "abandoned": attack.abandoned,
    }


def run_parallel(env, net, attack="state", horizon=20, maxlen=1000, budget=200,
                 num_traj=10, basis=2, basis_type="block", abandon=None,
                 warm_start=None, legacy=False, pgd_mode=None, target_space="continuous",
                 optimizer="racos", cem_pop=100, cem_elite=0.15, num_iter_override=None, seed=0, workers=None,
                 progress=True):
    'Attack num_traj trajectories across a process pool, returning them in order'

    workers = workers or min(num_traj, mp.cpu_count())

    jobs = [dict(env=env, net=net, attack=attack, horizon=horizon, maxlen=maxlen,
                 budget=budget, num_traj=num_traj, basis=basis, basis_type=basis_type,
                 abandon=abandon, warm_start=warm_start, legacy=legacy,
                 pgd_mode=pgd_mode, target_space=target_space, optimizer=optimizer,
                 cem_pop=cem_pop, cem_elite=cem_elite,
                 num_iter_override=num_iter_override, seed=seed,
                 index=i)
            for i in range(num_traj)]

    results = []
    # maxtasksperchild=1 keeps each MuJoCo instance in a fresh process
    pool = mp.Pool(processes=workers, maxtasksperchild=1)
    try:
        for r in pool.imap_unordered(_worker, jobs):
            results.append(r)
            if progress:
                print("  trajectory %2d done  reward %9.1f   (%d/%d)"
                      % (r["index"], r["traj"].reward, len(results), num_traj), flush=True)
    finally:
        pool.close()
        pool.join()

    results.sort(key=lambda r: r["index"])
    return ([r["traj"] for r in results],
            sum(r["evaluated"] for r in results),
            sum(r["abandoned"] for r in results))
