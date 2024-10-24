import numpy as np
import matplotlib
import random
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

from optimization_attack import OptimizationAttack

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Hopper.Hopper import Hopper
from envs.Ant.AntEnv import Ant

environment = HalfCheetah
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")
PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Time")
ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Time")
LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Time")

ENV = PPO_attack
traj = PPO_attack.adv_trajs[0]

current_time = 0
alt_time = 0
counter = 0
step = 0
constant_times = []
opt_times = []
alt_times = []
rand_times = []

while step < len(traj.adv_actions):
    
    start_time = (step%ENV.time_horizon + 1) * ENV.period
    if (step + 1) % ENV.time_horizon == 0:
        current_time = 0
        alt_time = 0
        counter += 1
    else:
        time_step = (start_time - current_time) + traj.timing[step] 
        current_time += time_step

        if step%2 == 0:
            alt_time += start_time - alt_time + 0.004
        else:
            alt_time += start_time - alt_time - 0.004

    opt_times.append(current_time + counter * ENV.period * ENV.time_horizon - 0.005)
    constant_times.append(step * ENV.period - 0.005)
    alt_times.append(max(alt_time + counter * ENV.period * ENV.time_horizon - 0.005, 0))
    rand_times.append(max(step * ENV.period + np.random.uniform(-0.005, 0.005) - 0.005, 0))
    
    step += 1

num_points = 19
offset = 0.01

plt.scatter(constant_times[1:num_points+1], [0.6]*(num_points), s=100, marker="v", label="Clean")
plt.scatter(alt_times[:num_points], [0.4]*num_points, s=100, marker="v", label="Alternating")
plt.scatter(rand_times[1:num_points+1], [0.2]*(num_points), s=100, marker="v", label="Random")
plt.scatter(opt_times[:num_points], [0]*num_points, s=100, marker="v", label="Optimized")

#plt.vlines(x=np.array(list(range(1,num_points + 2,1)))*0.01 - 0.01, ymin=-0.02, ymax=0.62, colors='black', ls='--', lw=1)
for x in [0, 0.2, 0.4, 0.6]:
    for y in np.array(range(0, 20, 1))*0.01:
        plt.arrow(y, x + 0.05 - offset, 0, -0.05, length_includes_head=True, width=0.0002, head_width=0.001, head_length=0.01, color="black")

plt.hlines(y = [0 - offset, 0.2 - offset, 0.4 - offset, 0.6 - offset], xmin=0, xmax=0.195, colors='black')
plt.xlabel("Activation Time")
plt.yticks([0, 0.2, 0.4, 0.6], ["Optimized\nReward:-630", "Random\nReward:4593", "Alternating\nReward:1737", "Clean\nReward:7322"])

plt.show()

'''
confirm_env = ENV.env_constructor(ENV.net_type).env
confirm_env.custom_env.env.seed(0)
done = False
env_attr = None
current_state = deepcopy(traj.start_state)
obs = confirm_env.reset(traj.start_state, env_attr)
step = 0
current_time = 0

while step < len(traj.adv_actions):
    #adv_obs = traj.adv_states[step]
    #print(traj.adv_states[step], obs)
    adv_obs = obs
    action = confirm_env.predict(adv_obs)

    #np.random.seed(0)
    #random.seed(0)
    #torch.manual_seed(0)
    
    if ENV.attack_type == "state":
        state, _, _, _ = confirm_env.step(action, change_filter=True)
    else:
        start_time = (step%ENV.time_horizon + 1) * ENV.period
        if (step + 1) % (ENV.time_horizon) == 0:
            time_step = start_time - current_time
            current_time = 0
        else:
            #time_step = start_time - current_time
            #time_step = (start_time - current_time) + traj.timing[step] 
            #time_step = start_time - current_time + np.random.uniform(-0.005, 0.005, 1)
            if step%2 == 0:
                time_step = start_time - current_time + 0.00499
            else:
                time_step = start_time - current_time - 0.00499
            current_time += time_step
        state, _, _, _ = confirm_env.step(action, time_step=time_step)

    obs = state[1]

    step += 1

    #confirm_env.custom_env.env.render()
    #time.sleep(0.01)
    
    #print(current_time)
    
print(confirm_env.custom_env.total_true_reward)
'''
