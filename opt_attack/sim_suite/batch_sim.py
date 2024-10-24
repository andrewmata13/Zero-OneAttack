import numpy as np

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

from optimization_attack import OptimizationAttack

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Hopper.Hopper import Hopper
from envs.Ant.AntEnv import Ant

'''
# Look-Ahead: 20-PPO, ATLA, LSTM
environment = HalfCheetah
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state =  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 20

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1)
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2)
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

# For Testing
#PPO_attack.generate(num_traj=1)
#PPO_attack.save_attack("Adv_Traj/Cheetah_PPO_Test")

#PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Test")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Cheetah_PPO_PPO_Simulations")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Cheetah_ATLA_Simulations")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Cheetah_LSTM_Simulations")

#PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Simulations")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[4], render=True)

#ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Simulations")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[1], render=True)


# Look-Ahead: 300-ATLA, 400-PPO, 600-LSTM
environment = Hopper
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = np.array([0,1.25,0,0,0,0,0,0,0,0,0,0])
eps = 0.075
num_iter = 2
step_size = 0.075
time_horizon = 400
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1)
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2)
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Hopper_PPO_Simulations")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Hopper_ATLA_Simulations")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Hopper_LSTM_Simulations")

#PPO_attack.load_attack("Adv_Traj/Hopper_PPO_Simulations")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Hopper_ATLA_Simulations")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Hopper_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)



# Look-Ahead: 350 PPO, ATLA, 700 for LSTM
environment = Walker2D
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state = np.array([0,1.25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.05
num_iter = 2
step_size = 0.05
time_horizon = 700
maxlen = 1000

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1)
ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2)
LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Walker_PPO_Simulations")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Walker_ATLA_Simulations")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Walker_LSTM_Simulations")

#PPO_attack.load_attack("Adv_Traj/Walker_PPO_Simulations")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Walker_ATLA_Simulations")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Walker_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)
'''

# Look-Ahead: 20 PPO, ATLA, LSTM
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

PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1)
#ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2)
#LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

# For Testing
PPO_attack.generate(num_traj=1)
PPO_attack.save_attack("Adv_Traj/Ant_PPO_Test")

PPO_attack.load_attack("Adv_Traj/Ant_PPO_Test")
PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Ant_PPO_Simulations")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Ant_ATLA_Simulations")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Ant_LSTM_Simulations")

#PPO_attack.load_attack("Adv_Traj/Ant_PPO_Simulations")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Ant_ATLA_Simulations")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Ant_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[9], render=True)

