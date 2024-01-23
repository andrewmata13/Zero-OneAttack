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

matplotlib.use('TkAgg') # set backend
p = 'bak_matplotlib.mlpstyle'
plt.style.use(['bmh', p])

environment = HalfCheetah
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state =  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.15
num_iter = 2
step_size = 0.15
#time_horizon = 20
maxlen = 1000

'''
rewards = []
times = []

for time_horizon in range(1, 106, 5):
    PPO_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type1, "state+time")
    a = time.perf_counter()
    PPO_attack.generate(num_traj=1)
    b = time.perf_counter()
    rewards.append(PPO_attack.adv_trajs[0].reward)
    times.append(b - a)
    del PPO_attack

print("Rewards: ", rewards)
print("Times: ", times)
'''



state_rewards =  [3606.5814141917341, 1484.5876266141368, -424.93218662476966, -636.2293341610327, -1343.255320016311, -1160.4651399767456, -1207.0836827022074, -1014.542202049926, -638.0509652976616, -565.5726812131434, -517.9309965074838, -611.445584668871, -556.3148627567183, -514.87588708678, -505.76818249777665, -478.1681044721679, -425.5282860158965, -386.8695598451518, -334.01759171817224, -351.8451721502925, -346.4532139986419]
#times = [66.84221197804436, 239.98531707422808, 419.7335611958988, 599.0536586898379, 764.2704953122884, 940.1938644489273, 1123.4220844041556, 1300.1438017943874, 1474.0806689038873, 1620.6680047069676, 1734.8771170210093, 1694.602504691109, 1722.0410100920126, 1747.7152990042232, 1761.3468014816754, 1757.4856208637357, 1732.348510463722, 1687.4246645369567, 1610.9171823463403, 1703.9166566631757, 1605.7536939447746]

time_rewards = [7237.76995336619, 1752.7880005040608, 506.74536909836655, 3.704385410983092, -70.73740498378673, -148.41715564581065, -186.97025972654018, -173.7997132155565, -111.67371461514053, -44.04943002299227, -189.99668106990066, -134.013811152157, -354.91055559499915, -168.45295071344833, -382.692803625107, -76.35562825053093, -326.2602698965017, -342.37778637584114, -201.10922001906727, -93.37033053759963, -250.28036704344944]

state_time_rewards =  [3766.026684384265, -994.4837713732883, -964.9861612718585, -1480.0742467344842, -2227.458577984886, -2388.239346873464, -1924.0637884510509, -1597.244458530472, -1426.5492256699317, -1423.8425116467633, -946.5159932649258, -892.5928774623796, -749.3274653691406, -810.6415200100756, -659.6327389998484, -711.3279295139615, -669.2479239448735, -619.3866697681237, -558.5657874459095, -532.4776385373335, -501.15115674701667]

plt.plot(list(range(1,106,5)), state_rewards, label="State")
#plt.plot(list(range(0,105,5)), [-530]*21, label="Zhang")
plt.plot(list(range(1,106,5)), time_rewards, label="Time")
plt.plot(list(range(1,106,5)), state_time_rewards, label="State+Time")
plt.title("Half Cheetah Optimization Parameter Search")
plt.ylabel("Average Reward")
plt.xlabel("Time Horizon")
plt.legend()
plt.show()




#ATLA_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type2)
#LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

#PPO_attack.generate(num_traj=10)
#PPO_attack.save_attack("Adv_Traj/Cheetah_PPO_Simulations")

#ATLA_attack.generate(num_traj=10)
#ATLA_attack.save_attack("Adv_Traj/Cheetah_ATLA_Simulations")

#LSTM_attack.generate(num_traj=10)
#LSTM_attack.save_attack("Adv_Traj/Cheetah_LSTM_Simulations")

#PPO_attack.load_attack("Adv_Traj/Cheetah_PPO_Simulations")
#PPO_attack.check_trajectory(PPO_attack.adv_trajs[0], render=True)

#ATLA_attack.load_attack("Adv_Traj/Cheetah_ATLA_Simulations")
#ATLA_attack.check_trajectory(ATLA_attack.adv_trajs[0], render=True)

#LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[1], render=True)

