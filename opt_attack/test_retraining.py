import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

from optimization_attack import OptimizationAttack
from other_attacks.optimal_attack.policy_gradients.models import CtsPolicy, CtsLSTMPolicy

from envs.HalfCheetah.CheetahEnv import HalfCheetah
from envs.Walker2D.Walker2DEnv import Walker2D
from envs.Hopper.Hopper import Hopper
from envs.Ant.AntEnv import Ant

# Look-Ahead: 20
environment = HalfCheetah
net_type1 = "PPO"
net_type2 = "ATLA"
net_type3 = "LSTM"
start_state =  np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
eps = 0.15
num_iter = 2
step_size = 0.15
time_horizon = 20
maxlen = 1000

LSTM_attack = OptimizationAttack(environment, start_state, eps, num_iter, step_size, time_horizon, maxlen, net_type3)

LSTM_attack.load_attack("Adv_Traj/Cheetah_LSTM_Simulations")
#LSTM_attack.check_trajectory(LSTM_attack.adv_trajs[0], render=True)

learning_rate = 1e-3

loss = nn.MSELoss
model = environment(net_type3).model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_reward = 0
'''
for trajectory in LSTM_attack.adv_trajs:
    env = environment(net_type3).env

    start_state = trajectory.start_state
    adv_states = trajectory.adv_states
    env_attr = None

    obs = env.reset(start_state, env_attr)

    model.reset()
    
    step = 0

    while step < len(adv_states):
        stateTensor = torch.tensor(adv_states[step]).type(torch.FloatTensor).unsqueeze(0)
        action_pds = model(stateTensor)
        action = torch.clamp(action_pds[0], min=-1, max=1).detach().numpy()
        
        state, reward, done, _ = env.step(action, change_filter=True)
        obs = state[1]

        step += 1

    total_reward += env.custom_env.total_true_reward

prev_avg = total_reward / len(LSTM_attack.adv_trajs)
print(prev_avg)
'''
#print([x for x in model.parameters()])
print("--------------------------------------------------------------------------------------")

# Re-Training
model.train()

for trajectory in LSTM_attack.adv_trajs:
    env = environment(net_type3).env

    start_state = trajectory.start_state
    adv_states = trajectory.adv_states
    env_attr = None

    obs = env.reset(start_state, env_attr)

    model.reset()
    
    step = 0

    total_reward = 0
    prev_reward = 0

    while step < len(adv_states):
        stateTensor = torch.tensor(adv_states[step]).type(torch.FloatTensor).unsqueeze(0)
        action_pds = model(stateTensor)
        action = torch.clamp(action_pds[0], min=-1, max=1).detach().numpy()
        
        state, reward, done, _ = env.step(action, change_filter=True)
        obs = state[1]

        if step % 50 == 0:
            total_reward = env.custom_env.total_true_reward - prev_reward
            prev_reward = env.custom_env.total_true_reward
            loss = torch.tensor([total_reward]).type(torch.FloatTensor)
            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_reward = 0
            
        step += 1

#print([x for x in model.parameters()])

model.eval()

total_reward = 0

for trajectory in LSTM_attack.adv_trajs:
    env = environment(net_type3).env

    start_state = trajectory.start_state
    adv_states = trajectory.adv_states
    env_attr = None

    obs = env.reset(start_state, env_attr)

    model.reset()
    
    step = 0

    while step < len(adv_states):
        stateTensor = torch.tensor(adv_states[step]).type(torch.FloatTensor).unsqueeze(0)
        action_pds = model(stateTensor)
        action = torch.clamp(action_pds[0], min=-1, max=1).detach().numpy()
        
        state, reward, done, _ = env.step(action, change_filter=True)
        obs = state[1]

        step += 1

    total_reward += env.custom_env.total_true_reward

prev_avg = total_reward / len(LSTM_attack.adv_trajs)
print(prev_avg)
