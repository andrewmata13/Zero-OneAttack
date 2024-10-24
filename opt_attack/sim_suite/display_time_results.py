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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Time")
print("Average Half Cheetah PPO Random Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Half Cheetah PPO Alternating Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Half Cheetah PPO Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Time")
print("Average Half Cheetah ATLA Random Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Half Cheetah ATLA Alternating Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Half Cheetah ATLA Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Time")
print("Average Half Cheetah LSTM Random Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Half Cheetah LSTM Alternating Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Half Cheetah LSTM Time Attack Reward:", LSTM_attack.avg_reward())

environment = Hopper
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.075
num_iter = 2
step_size = 0.075
time_horizon = 1000
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

PPO_attack.load_attack("Adv_Traj/Hopper_PPO_Time")
print("Average Hopper PPO Random Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Hopper PPO Alternating Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Hopper PPO Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Hopper_ATLA_Time")
print("Average Hopper ATLA Random Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Hopper ATLA Alternating Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Hopper ATLA Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Hopper_LSTM_Time")
print("Average Hopper LSTM Random Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Hopper LSTM Alternating Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Hopper LSTM Time Attack Reward:", LSTM_attack.avg_reward())

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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

PPO_attack.load_attack("Adv_Traj/Walker_PPO_Time")
print("Average Walker PPO Random Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Walker PPO Alternating Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Walker PPO Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Walker_ATLA_Time")
print("Average Walker ATLA Random Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Walker ATLA Alternating Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Walker ATLA Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Walker_LSTM_Time")
print("Average Walker LSTM Random Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Walker LSTM Alternating Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Walker LSTM Time Attack Reward:", LSTM_attack.avg_reward())

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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

PPO_attack.load_attack("Adv_Traj/Ant_PPO_Time")
print("Average Ant PPO Random Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Ant PPO Alternating Reward:", PPO_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Ant PPO Time Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Ant_ATLA_Time")
print("Average Ant ATLA Random Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Ant ATLA Alternating Reward:", ATLA_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Ant ATLA Time Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Ant_LSTM_Time")
print("Average Ant LSTM Random Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Random"))
print("Average Ant LSTM Alternating Reward:", LSTM_attack.time_attack_reward(num_traj=10, attack_type="Alternating"))
print("Average Ant LSTM Time Attack Reward:", LSTM_attack.avg_reward())

