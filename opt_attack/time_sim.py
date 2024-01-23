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
start_state =  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 100

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

#PPO_attack.generate(num_traj=1)
#PPO_attack.save_attack("Adv_Traj/Cheetah_Test")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Cheetah_ATLA_Time")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Cheetah_LSTM_Time")

#PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Time")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Time")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Time")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)


environment = Hopper
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = np.array([0,1.25,0,0,0,0,0,0,0,0,0,0])
eps = 0.075
num_iter = 2
step_size = 0.075
time_horizon = 1000
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Hopper_PPO_Time")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Hopper_ATLA_Time")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Hopper_LSTM_Time")

#PPO_attack.load_attack("Adv_Traj/Hopper_PPO_Time")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Hopper_ATLA_Time")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Hopper_LSTM_Time")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)



environment = Walker2D
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = np.array([0,1.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.05
num_iter = 2
step_size = 0.05
time_horizon = 1000
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Walker_PPO_Time")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Walker_ATLA_Time")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Walker_LSTM_Time")

#PPO_attack.load_attack("Adv_Traj/Walker_PPO_Time")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Walker_ATLA_Time")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Walker_LSTM_Time")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)


environment = Ant
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = np.array([0,0,0.75,1] + [0]*109)
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "time")
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2, "time")
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3, "time")

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Ant_PPO_Time")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Ant_ATLA_Time")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Ant_LSTM_Time")

#PPO_attack.load_attack("Adv_Traj/Ant_PPO_Time")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Ant_ATLA_Time")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Ant_LSTM_Time")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)
