import numpy as np

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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state+time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state+time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state+time")

PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_State+Time")
print("Average Half Cheetah PPO State+Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_State+Time")
print("Average Half Cheetah ATLA State+Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_State+Time")
print("Average Half Cheetah LSTM State+Time Attack Reward:", LSTM_attack.avg_reward())

environment = Hopper
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.075
num_iter = 2
step_s2ize = 0.075
time_horizon = 1000
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state+time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state+time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state+time")

PPO_attack.load_attack("Adv_Traj/Hopper_PPO_State+Time")
print("Average Hopper PPO State+Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Hopper_ATLA_State+Time")
print("Average Hopper ATLA State+Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Hopper_LSTM_State+Time")
print("Average Hopper LSTM State+Time Attack Reward:", LSTM_attack.avg_reward())

environment = Walker2D
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.05
num_iter = 2
step_size = 0.05
time_horizon = 1000
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state+time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state+time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state+time")

PPO_attack.load_attack("Adv_Traj/Walker_PPO_State+Time")
print("Average Walker PPO State+Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Walker_ATLA_State+Time")
print("Average Walker ATLA State+Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Walker_LSTM_State+Time")
print("Average Walker LSTM State+Time Attack Reward:", LSTM_attack.avg_reward())

environment = Ant
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state+time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state+time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state+time")

PPO_attack.load_attack("Adv_Traj/Ant_PPO_State+Time")
print("Average Ant PPO State+Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Ant_ATLA_State+Time")
print("Average Ant ATLA State+Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Ant_LSTM_State+Time")
print("Average Ant LSTM State+Time Attack Reward:", LSTM_attack.avg_reward())

