import gym
import time
import pickle
import sys
import os
import random
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from graphviz import Digraph
from copy import deepcopy

import mujoco_py

sys.path.insert(0,'..')
sys.path.insert(1,'../..')

from other_attacks.optimal_attack.policy_gradients.models import CtsPolicy, CtsLSTMPolicy

class ExtendedHopper():
    def __init__(self, network, custom_env, attack_model, eps):
        self.network = network
        self.custom_env = custom_env
        self.attack_model = attack_model
        self.eps = eps
        
    def reset(self, uState, attributes=None):
        state = self.custom_env.reset(uState, attributes, name="Hopper")
        return state

    def _get_obs(self):
        qpos = self.custom_env.env.sim.data.qpos
        qvel = self.custom_env.env.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def step(self, action, change_filter=False, time_step=None):
        return self.custom_env.step(action, change_filter, name="Hopper", time_step=time_step)

    def predict(self, state):
        stateTensor = torch.tensor(state).type(torch.FloatTensor).unsqueeze(0)
        action_pds = self.network(stateTensor)
        return torch.clamp(action_pds[0], min=-1, max=1).detach().numpy()

    def opt_attack(self, obs):
        adv_perturbation_pds = self.attack_model(torch.tensor([obs]).type(torch.FloatTensor))
        next_adv_perturbations = self.attack_model.sample(adv_perturbation_pds)
        adv_obs = obs + torch.tanh(next_adv_perturbations).numpy()[0] * self.eps
        return adv_obs

class Hopper:
    def __init__(self, model="PPO"):
        self.bounds = [
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5],
            [-5, 5]
       ]

        if model == "PPO":
            self.checkpoint = torch.load("../envs/Hopper/Hopper_PPO.model")
            self.attack_checkpoint = torch.load("../envs/Hopper/Hopper_Attack_PPO.model")            
            self.model = CtsPolicy(11, 3, "orthogonal")
            self.attack_model = CtsPolicy(11, 11, "orthogonal")
        elif model == "ATLA":
            self.checkpoint = torch.load("../envs/Hopper/Hopper_ATLA.model")
            self.attack_checkpoint = torch.load("../envs/Hopper/Hopper_Attack_ATLA.model")
            self.model = CtsPolicy(11, 3, "orthogonal")
            self.attack_model = CtsPolicy(11, 11, "orthogonal")
        elif model == "LSTM":
            self.checkpoint = torch.load("../envs/Hopper/Hopper_LSTM.model")
            self.attack_checkpoint = torch.load("../envs/Hopper/Hopper_Attack_LSTM.model")
            self.model = CtsLSTMPolicy(11, 3, "orthogonal")
            self.attack_model = CtsLSTMPolicy(11, 11, "orthogonal")
        else:
            exit("Enter valid model choice(PPO or ATLA)")
            
        #Load model
        self.model.load_state_dict(self.checkpoint['policy_model'])
        self.model.log_stdev.data[:] = -100
        self.model.eval()

        #Load Attack Model
        self.attack_model.load_state_dict(self.attack_checkpoint['adversary_policy_model'])
        self.attack_model.log_stdev.data[:] = -100
        self.attack_model.eval()

        #Noise Level
        self.eps = 0.075
        self.period = 0.01
        
        #Load environment
        self.custom_env = self.checkpoint['envs'][0]
        self.env = ExtendedHopper(self.model, self.custom_env, self.attack_model, self.eps)

        self.actionBounds = [[-1, 1], [-1, 1], [-1, 1]]
        self.name = "Hopper"
        self.usesCustom = True


