import time
import pickle
import sys
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from numpy.linalg import norm
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from copy import deepcopy
import os

from scipy.optimize import differential_evolution, dual_annealing, brute, minimize
from advertorch.attacks import PGDAttack, LinfPGDAttack, L2PGDAttack
import zoopt
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

sys.path.insert(0,'..')
sys.path.insert(1,'../..')

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class OptimizationAttack:
    def __init__(self, environment, start_state, eps, num_iter=15, step_size=0.01, time_horizon=20, maxlen=1000, net_type=None, attack_type="state"):
        self.env_constructor = environment
        self.net_type = net_type
        
        # Load Environment
        self.environment = environment(net_type)
        self.env = self.environment.env
        self.name = self.environment.name
        self.bounds = self.environment.bounds
        self.start_state = start_state
        self.action_bounds = self.environment.actionBounds
        self.num_actions = len(self.environment.actionBounds)
        self.period = self.environment.period
        
        # Load PGD Attack Model
        self.pgdModel = AdvertorchAdapter(environment(net_type).model)
        self.pgdModel.eval();

        if net_type == "LSTM":
            self.pgdModel.net.detach_hidden = True
            self.pgdModel.net.hidden = [torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)]
            self.current_hidden = None
            
        # Hyper-Parameters
        self.eps = eps
        self.num_iter = num_iter
        self.step_size = step_size
        self.time_horizon = time_horizon
        self.maxlen = maxlen

        # Initialize Adversarial Trajectories
        self.adv_trajs = []

        # Attack Types: "state", "time", "state+time"
        self.attack_type = attack_type

    def optimize(self, current_state, env_attr):
        # Keep Track of Best Actions/Rewards Found
        self.best_obs = None
        self.best_reward = 10000
        self.best_action = None
        self.numSim = 0
        self.done = False
        self.best_delay = None

        # Reset Environment
        if self.environment.usesCustom:
            obs = self.env.reset(current_state, env_attr)
        else:
            obs = self.env.reset(current_state)

        # Find Initial Network Predicted Actions With No Time Delays
        done = False
        u0 = []
        for _ in range(self.time_horizon):
            next_action = self.env.predict(obs)
            state, _, _, _ = self.env.step(next_action)
            if self.environment.usesCustom:
                obs = state[1]
            else:
                obs = state
            u0.append(next_action)
        u0 = np.array(u0).flatten()

        def cost_function(u):
            if isinstance(u, zoopt.solution.Solution):
                u = u.get_x()
                
            # Reset Environment
            if self.environment.usesCustom:
                obs = self.env.reset(current_state, env_attr)
            else:
                obs = self.env.reset(current_state)

            total_reward = 0
            observations = []
            actions = []

            # Reset Model Hidden State to Beginning Hidden State
            if self.net_type == "LSTM":
                self.env.network.hidden = deepcopy(self.current_hidden)
            
            # Initialize PGD Attack
            attack = PGDAttack(self.pgdModel, eps=self.eps, nb_iter=self.num_iter, eps_iter=self.step_size, clip_min=-1000, clip_max=1000)

            # If present, separate timing params from action params
            opt_actions = []
            timing = []
            if self.attack_type == "state+time":
                for i, param in enumerate(u):
                    if i%(self.num_actions + 1) == 0:
                        timing.append(param)
                    else:
                        opt_actions.append(param)
            elif self.attack_type == "state":
                opt_actions = u
            else:
                timing = u
                
            current_time = 0
                
            for i in range(self.time_horizon):
                # For LSTM, set hidden state for PGD Model to Network hidden state
                if self.net_type == "LSTM":
                    self.pgdModel.net.hidden = deepcopy([self.env.network.hidden[0].detach(), self.env.network.hidden[1].detach()])

                # If only timing attack, do not do pgd attack on state input
                if self.attack_type == "time":
                    adv_obs = obs
                else:
                    # Set Up Target for PGD Attack and Do Attack
                    target = torch.tensor(np.array([opt_actions[i*self.num_actions:i*self.num_actions + self.num_actions]])).type(torch.FloatTensor)
                    torch_state = torch.tensor(obs).type(torch.FloatTensor)
                    
                    Timers.tic("PGD Attack")
                    output = attack.perturb(torch_state.unsqueeze(0), y=target)
                    Timers.toc("PGD Attack")

                    # Collect Adversarial Observation and Action
                    adv_obs = output.detach().numpy()[0]

                Timers.tic("Network Execution")
                adv_action = self.env.predict(adv_obs)
                Timers.toc("Network Execution")

                observations.append(adv_obs)
                actions.append(adv_action)

                Timers.tic("Environment Step")
                # Take Environment Step
                if self.attack_type == "state":
                    state, reward, done, _ = self.env.step(adv_action)
                else:
                    start_time = (i+1) * self.period
                    if i == self.time_horizon - 1:
                        time_step = start_time - current_time
                    else:
                        time_step = (start_time - current_time) + timing[i] 
                    state, reward, done, _ = self.env.step(adv_action, time_step=time_step)
                    current_time += time_step

                Timers.toc("Environment Step")

                if done:
                    self.done = True
                    break
                
                if self.environment.usesCustom:
                    obs = state[1]
                else:
                    obs = state

                total_reward += reward

            if self.environment.usesCustom:
                total_reward = self.env.custom_env.total_true_reward

            # Keep Track of Best Found Reward
            if total_reward < self.best_reward and not self.done:
                if self.name == "Ant" or self.name == "Half Cheetah":
                    # For non-safety critical envs, simulate forward to see if the environment will reach a "bad" state
                    # (done condition = True) and try to avoid those (will allow us to find the backwards-moving behaviors) 
                    flag = False
                    for i in range(self.time_horizon):
                        action = self.env.predict(obs)
                        state, reward, done, _ = self.env.step(action)
                        obs = state[1]
                        if done:
                            flag = True

                    if not flag or self.best_action is None:
                        self.best_reward = total_reward
                        self.best_obs = observations
                        self.best_action = actions
                        self.best_delay = timing
                elif self.name == "Hopper" or self.name == "Walker2D":
                    if self.best_action is None:
                        self.best_reward = total_reward
                        self.best_obs = observations
                        self.best_action = actions
                        self.best_delay = timing
            elif total_reward < self.best_reward and self.done:
                if self.name == "Hopper" or self.name == "Walker2D":
                    self.best_reward = total_reward
                    self.best_obs = observations
                    self.best_action = actions
                    self.best_delay = timing
                    print("Found Terminal Trace")
                    
                    # Add additional reward constant for the optimizer for unsafe traces to encourage
                    # finding unsafe traces for the safety critical envs (Hopper and Walker2D)
                    total_reward -= 1000
                elif self.name == "Ant" or self.name == "Half Cheetah":
                    if self.best_action is None:
                        self.best_obs = observations
                        self.best_action = actions
                        self.best_delay = timing
                    
            self.done = False
            self.numSim += 1
            
            #print(total_reward)
            return total_reward

        '''
        # Calculate Bounds for Action Search Space
        bounds = []
        for _ in range(int(len(u0)/self.num_actions)):
            bounds += self.action_bounds
        bounds = np.array(bounds)

        # Scipy Global Optimizer
        Timers.tic("Global Optimization")
        result = differential_evolution(cost_function, bounds, x0=u0, maxiter=1, popsize=5, polish=False)
        print("Ideal Reward", result.fun)
        print("Number Simulations", self.numSim)
        Timers.toc("Global Optimization")

        # Random Simulations
        Timers.tic("Global Optimization")
        simulations = 5000
        for _ in range(simulations):
            actions = np.random.uniform(low=bounds[:,0], high=bounds[:,1])
            rew = cost_function(actions)
            if self.done and (self.name == "Walker" or self.name == "Hopper"):
                break

        print("Best Found:", self.best_reward)
        Timers.toc("Global Optimization")
        '''

        # Zeroth Order Optimization
        Timers.tic("Global Optimization")
        if self.attack_type == "state":
            dim_size = self.num_actions * self.time_horizon
            dim = Dimension(dim_size, [[-1, 1]]*dim_size, [True]*dim_size)
            obj = Objective(cost_function, dim)
            solution = Opt.min(obj, Parameter(budget=min(5000, 17 * dim_size)))
        elif self.attack_type == "time":
            dim_size = self.time_horizon
            bounds = []
            for i in range(dim_size):
                bounds.append([-self.period/2, self.period/2])

            dim = Dimension(dim_size, bounds, [True]*dim_size)
            obj = Objective(cost_function, dim)
            solution = Opt.min(obj, Parameter(budget=min(5000, 17 * dim_size)))
        elif self.attack_type == "state+time":
            dim_size = (self.num_actions + 1) * self.time_horizon
            bounds = []
            for i in range(dim_size):
                if i % (self.num_actions + 1) == 0:
                    bounds.append([-self.period/2, self.period/2])
                else:
                    bounds.append([-1,1])

            dim = Dimension(dim_size, bounds, [True]*dim_size)
            obj = Objective(cost_function, dim)
            solution = Opt.min(obj, Parameter(budget=min(5000, 17 * dim_size)))
        else:
            exit("Invalid attack type specified")

            
        Timers.toc("Global Optimization")

        return self.best_action, self.best_obs, self.done, self.best_delay
            
            
    def next_adv(self, current_state, env_attr):
        # Get Next Adversarial Actions/States
        actions, adv_states, done, timing = self.optimize(current_state, env_attr)

        # Reset Environment
        if self.environment.usesCustom:
            obs = self.env.reset(current_state, env_attr)
        else:
            obs = self.env.reset(current_state)

        if self.net_type == "LSTM":
            self.env.network.hidden = deepcopy(self.current_hidden)

        current_time = 0
            
        for i in range(len(actions)):
            # Compute Noise to Check within Noise Cap
            adv = adv_states[i]
            noise = calculateError(np.array(obs), adv, self.bounds)

            print("Noise Found:", noise)

            if self.attack_type == "state":
                assert noise <= self.eps + 1e-2

            action = self.env.predict(adv)

            assert actions[i][0][0] == action[0][0]

            if self.attack_type == "state":
                state, reward, _, _ = self.env.step(action)
            else:
                start_time = (i+1) * self.period
                if i == self.time_horizon - 1:
                    time_step = start_time - current_time
                else:
                    time_step = (start_time - current_time) + timing[i] 
                state, reward, done, _ = self.env.step(action, time_step=time_step)
                current_time += time_step
            
            if self.environment.usesCustom:
                obs = state[1]
            else:
                obs = state
        
        print("Total Reward:", self.env.custom_env.total_true_reward)
        
        return state, adv_states, actions, done, timing
        
    def generate(self, num_traj=1):
        #Timers.tic("Top")
        test_env = self.env_constructor(self.net_type).env
        test_env.custom_env.env.seed(num_traj)
        starting_states = []

        # If more than one trajectory, create random starting states
        if num_traj > 1:
            for _ in range(num_traj):
                if self.name == "Ant":
                    starting_states.append(np.concatenate((np.array([0, 0]), np.array(test_env.custom_env.env.reset()))))
                else:
                    starting_states.append(np.concatenate((np.array([0]), np.array(test_env.custom_env.env.reset()))))
        else:
            starting_states.append(self.start_state)

        self.current_time = 0
            
        for state in starting_states:
            # Environment For Confirming Trajectory
            test_env = self.env_constructor(self.net_type).env
            test_env.custom_env.env.seed(num_traj)

            self.start_state = state
                
            # Initialize Start State and Environments
            done = False
            env_attr = None
            current_state = deepcopy(self.start_state)

            obs = test_env.reset(self.start_state)

            # Collect Adversarial States and Actions
            all_states = []
            all_actions = []
            all_timing = []

            for i in range(int(self.maxlen / self.time_horizon)):
                if self.net_type == "LSTM":
                    self.current_hidden = deepcopy([test_env.network.hidden[0].detach(), test_env.network.hidden[1].detach()])
                    
                print("Attack Iteration:", i)
                state, adv_states, adv_actions, done, timing = self.next_adv(current_state, env_attr)

                if self.environment.usesCustom:
                    env_attr = state[2:]
                    current_state = state[0]
                else:
                    current_state = state

                all_states += list(adv_states)
                all_actions += list(adv_actions)
                all_timing += list(timing)
                
                if not self.environment.usesCustom:
                    continue
                else:
                    # Rerun on Confirmation Environment
                    current_time = 0

                    input_state = adv_states[0]

                    for i in range(len(adv_actions)):
                        action = test_env.predict(adv_states[i])
                        
                        if self.attack_type == "state":
                            test_state, _, _, _ = test_env.step(action, change_filter=True)
                        else:
                            start_time = (i+1) * self.period
                            if i == self.time_horizon - 1:
                                time_step = start_time - current_time
                            else:
                                time_step = (start_time - current_time) + timing[i] 

                            test_state, _, _, _ = test_env.step(action, change_filter=True, time_step=time_step)
                            current_time += time_step

                        input_state = test_state[1]
                               
                    print("REWARD:", test_env.custom_env.total_true_reward)
                    
                    self.env = self.env_constructor(self.net_type).env
                    self.env.reset(test_state[0], test_state[2:])
                    current_state = deepcopy(test_state[0])
                    if self.net_type == "LSTM":
                        self.env.network.hidden = deepcopy([test_env.network.hidden[0].detach(), test_env.network.hidden[1].detach()])

                # Done Condition For Walker and Hopper, stop simulation if met
                if done and (self.name == "Walker" or self.name == "Hopper"):
                    break

            # Add Adversarial Trajectory
            traj = AdversarialTrajectory(self.start_state, all_states, all_actions, all_timing, self.env.custom_env.total_true_reward)
            self.adv_trajs.append(traj)
                
        #Timers.toc("Top")
        #Timers.print_stats()
            

    def check_trajectory(self, traj, render=False):
        confirm_env = self.env_constructor(self.net_type).env
        confirm_env.custom_env.env.seed(0)
        done = False
        env_attr = None
        current_state = deepcopy(traj.start_state)
        obs = confirm_env.reset(traj.start_state, env_attr)
        step = 0
        current_time = 0
        
        while step < len(traj.adv_actions):
            adv_obs = traj.adv_states[step]

            action = confirm_env.predict(adv_obs)

            print("Iteration:", step)
            print("Action:", action)
            print("Noise:", calculateError(obs, adv_obs, self.bounds))

            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)

            if self.attack_type == "state":
                state, _, _, _ = confirm_env.step(action, change_filter=True)
            else:
                start_time = (step%self.time_horizon + 1) * self.period
                if (step + 1) % (self.time_horizon) == 0:
                    time_step = start_time - current_time
                    current_time = 0
                else:
                    time_step = (start_time - current_time) + traj.timing[step] 
                    current_time += time_step
                state, _, _, _ = confirm_env.step(action, change_filter=True, time_step=time_step)

            #print(time_step)
                
            obs = state[1]

            if render:
                confirm_env.custom_env.env.render()
                time.sleep(0.001)

            #if step == 999:
            #    confirm_env.custom_env.env.render()
            #    time.sleep(120)
                
            step += 1

        print("Final Reward:", confirm_env.custom_env.total_true_reward)

    def avg_reward(self):
        if self.adv_trajs == None:
            exit("Compute Adversarial Trajectories Before Calling")

        total_rewards = 0

        rewards = []
        
        for i, traj in enumerate(self.adv_trajs):
            confirm_env = self.env_constructor(self.net_type).env
            confirm_env.custom_env.env.seed(0)
            done = False
            env_attr = None
            current_state = deepcopy(traj.start_state)
            obs = confirm_env.reset(traj.start_state, env_attr)
            step = 0
            current_time = 0

            while step < len(traj.adv_actions):
                adv_obs = traj.adv_states[step]
                action = confirm_env.predict(adv_obs)

                np.random.seed(0)
                random.seed(0)
                torch.manual_seed(0)

                if self.attack_type == "state":
                    state, _, _, _ = confirm_env.step(action, change_filter=True)
                else:
                    start_time = (step%self.time_horizon + 1) * self.period
                    if (step + 1) % (self.time_horizon) == 0:
                        time_step = start_time - current_time
                        current_time = 0
                    else:
                        time_step = (start_time - current_time) + traj.timing[step] 
                        current_time += time_step
                    state, _, _, _ = confirm_env.step(action, change_filter=True, time_step=time_step)

                obs = state[1]

                step += 1

            total_rewards += confirm_env.custom_env.total_true_reward
            rewards.append(confirm_env.custom_env.total_true_reward)
            
            #print(confirm_env.custom_env.total_true_reward)
            
        return total_rewards / len(self.adv_trajs), np.std(rewards)

    def optimal_attack_reward(self, num_traj=1):
        test_env = self.env_constructor(self.net_type).env
        test_env.custom_env.env.seed(num_traj)
        starting_states = []
        
        if num_traj > 1:
            for _ in range(num_traj):
                if self.name == "Ant":
                    starting_states.append(np.concatenate((np.array([0, 0]), np.array(test_env.custom_env.env.reset()))))
                else:
                    starting_states.append(np.concatenate((np.array([0]), np.array(test_env.custom_env.env.reset()))))
        else:
            starting_states.append(self.start_state)

        total_reward = 0
        rewards = []
            
        for state in starting_states:
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            test_env = self.env_constructor(self.net_type).env
            test_env.custom_env.env.seed(0)
            
            self.start_state = state
                
            # Initialize Start State and Environments
            done = False
            env_attr = None
            current_state = deepcopy(self.start_state)
            obs = test_env.reset(self.start_state)
            step = 0

            while step < 1000:
                #adv_obs = obs
                adv_obs = test_env.opt_attack(obs)       

                action = test_env.predict(adv_obs)

                state, reward, done, _ = test_env.step(action, change_filter=True)
                obs = state[1]

                step += 1

                if done and (self.name == "Hopper" or self.name == "Walker2D"):
                    break

                #test_env.custom_env.env.render()
                #time.sleep(0.001)

                #if step == 999:
                #    test_env.custom_env.env.render()
                #    time.sleep(120)
                
                
            total_reward += test_env.custom_env.total_true_reward
            rewards.append(test_env.custom_env.total_true_reward)

        return total_reward / num_traj, np.std(rewards)

    def time_attack_reward(self, num_traj=1, attack_type="Random"):
        # attack types: "Random, Alternating"
        test_env = self.env_constructor(self.net_type).env
        test_env.custom_env.env.seed(num_traj)
        starting_states = []
        
        if num_traj > 1:
            for _ in range(num_traj):
                if self.name == "Ant":
                    starting_states.append(np.concatenate((np.array([0, 0]), np.array(test_env.custom_env.env.reset()))))
                else:
                    starting_states.append(np.concatenate((np.array([0]), np.array(test_env.custom_env.env.reset()))))
        else:
            starting_states.append(self.start_state)

        total_reward = 0
        rewards = []
            
        for state in starting_states:
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            test_env = self.env_constructor(self.net_type).env
            test_env.custom_env.env.seed(0)
            
            self.start_state = state
                
            # Initialize Start State and Environments
            done = False
            env_attr = None
            current_state = deepcopy(self.start_state)
            obs = test_env.reset(self.start_state)
            step = 0
            current_time = 0

            while step < 1000:
                action = test_env.predict(obs)

                start_time = (step%self.time_horizon + 1) * self.period
                if (step + 1) % (self.time_horizon) == 0:
                    time_step = start_time - current_time
                    current_time = 0
                else:
                    if attack_type == "Random":
                        time_step = start_time - current_time + np.random.uniform(-0.005, 0.005, 1)
                    elif attack_type == "Alternating":
                        if step%2 == 0:
                            time_step = start_time - current_time + 0.00499
                        else:
                            time_step = start_time - current_time - 0.00499
                    else:
                        exit("Invalid Time Attack Option")
                        
                    current_time += time_step

                state, _, _, _ = test_env.step(action, change_filter=True, time_step=time_step)

                
                obs = state[1]

                step += 1

                if done and (self.name == "Hopper" or self.name == "Walker2D"):
                    break

                
            total_reward += test_env.custom_env.total_true_reward
            rewards.append(test_env.custom_env.total_true_reward)

        return total_reward / num_traj, np.std(rewards)
    
    def save_attack(self, filename):
        with open(filename, 'wb') as thefile:
            pickle.dump(self.adv_trajs, thefile)
        
    def load_attack(self, filename):
        with open(filename, 'rb') as thefile:
            self.adv_trajs += list(pickle.load(thefile))



