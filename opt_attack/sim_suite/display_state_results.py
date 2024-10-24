import numpy as np

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

from optimization_attack import OptimizationAttack

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Hopper.Hopper import Hopper
from envs.Ant.AntEnv import Ant

# Look-Ahead: 20-PPO, ATLA, LSTM
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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state")

PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Simulations")
print("Average Half Cheetah PPO Optimal Attack Reward:", PPO_attack.optimal_attack_reward(num_traj=10))
print("Average Half Cheetah PPO Optimization Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Simulations")
print("Average Half Cheetah ATLA Optimal Attack Reward:", ATLA_attack.optimal_attack_reward(num_traj=10))
print("Average Half Cheetah ATLA Optimization Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Simulations")
print("Average Half Cheetah LSTM Optimal Attack Reward:", LSTM_attack.optimal_attack_reward(num_traj=10))
print("Average Half Cheetah LSTM Optimization Attack Reward:", LSTM_attack.avg_reward())

# Look-Ahead: 300-ATLA, 400-PPO, 600-LSTM
environment = Hopper
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.075
num_iter = 2
step_size = 0.075
time_horizon = 300
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state")

PPO_attack.load_attack("Adv_Traj/Hopper_PPO_Simulations")
print("Average Hopper PPO Optimal Attack Reward:", PPO_attack.optimal_attack_reward(num_traj=10))
print("Average Hopper PPO Optimization Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Hopper_ATLA_Simulations")
print("Average Hopper ATLA Optimal Attack Reward:", ATLA_attack.optimal_attack_reward(num_traj=10))
print("Average Hopper ATLA Optimization Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Hopper_LSTM_Simulations")
print("Average Hopper LSTM Optimal Attack Reward:", LSTM_attack.optimal_attack_reward(num_traj=10))
print("Average Hopper LSTM Optimization Attack Reward:", LSTM_attack.avg_reward())

# Look-Ahead: 350-PPO, ATLA, 700-LSTM
environment = Walker2D
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = None
eps = 0.05
num_iter = 2
step_size = 0.05
time_horizon = 350
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state")

PPO_attack.load_attack("Adv_Traj/Walker_PPO_Simulations")
print("Average Walker PPO Optimal Attack Reward:", PPO_attack.optimal_attack_reward(num_traj=10))
print("Average Walker PPO Optimization Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Walker_ATLA_Simulations")
print("Average Walker ATLA Optimal Attack Reward:", ATLA_attack.optimal_attack_reward(num_traj=10))
print("Average Walker ATLA Optimization Attack Reward:", ATLA_attack.avg_reward())


LSTM_attack.load_attack("Adv_Traj/Walker_LSTM_Simulations")
print("Average Walker LSTM Optimal Attack Reward:", LSTM_attack.optimal_attack_reward(num_traj=10))
print("Average Walker LSTM Optimization Attack Reward:", LSTM_attack.avg_reward())


# Look-Ahead: 20 PPO, ATLA, LSTM
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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "state")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "state")

PPO_attack.load_attack("Adv_Traj/Ant_PPO_Simulations")
print("Average Ant PPO Optimal Attack Reward:", PPO_attack.optimal_attack_reward(num_traj=10))
print("Average Ant PPO Optimization Attack Reward:", PPO_attack.avg_reward())

ATLA_attack.load_attack("Adv_Traj/Ant_ATLA_Simulations")
print("Average Ant ATLA Optimal Attack Reward:", ATLA_attack.optimal_attack_reward(num_traj=10))
print("Average Ant ATLA Optimization Attack Reward:", ATLA_attack.avg_reward())

LSTM_attack.load_attack("Adv_Traj/Ant_LSTM_Simulations")
print("Average Ant LSTM Optimal Attack Reward:", LSTM_attack.optimal_attack_reward(num_traj=10))
print("Average Ant LSTM Optimization Attack Reward:", LSTM_attack.avg_reward())

