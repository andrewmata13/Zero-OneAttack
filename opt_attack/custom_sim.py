import numpy as np

from optimization_attack import OptimizationAttack

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Hopper.Hopper import Hopper
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Ant.AntEnv import Ant

# Parameters you can change
environments = [HalfCheetah, Hopper, Walker2D, Ant]
net_types = ["PPO", "ATLA", "LSTM"]
attack_types = ["state", "time", "state+time"]
num_traj = 10

# Important: Number of simulations for zeroth order optimizer, will dictate runtime heavily
attack_budget = 10

# Will contain the adv trajectories where the keys are (env, network type, attack type)
adversarial_trajectories = {}

for environment in environments:
    for net_type in net_types:
        params = environment(net_type).params

        if net_type == "PPO":
            time_horizon = params["time_horizon_PPO"]
        elif net_type == "ATLA":
            time_horizon = params["time_horizon_ATLA"]
        elif net_type == "LSTM":
            time_horizon = params["time_horizon_LSTM"]
            
        for attack_type in attack_types:
            attack_obj = OptimizationAttack(environment=environment,
                                            start_state=params["start_state"],
                                            eps=params["eps"],
                                            num_iter=params["num_iter"],
                                            step_size=params["step_size"],
                                            time_horizon=time_horizon,
                                            maxlen=params["maxlen"],
                                            net_type=net_type,
                                            attack_type=attack_type,
                                            attack_budget=attack_budget
            )
            attack_obj.generate(num_traj=num_traj)
            adversarial_trajectories[(environment, net_type, attack_type)] = attack_obj.adv_trajs
