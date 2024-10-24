import gym
import torch
import numpy as np

from other_attacks.optimal_attack.policy_gradients.models import CtsPolicy, CtsLSTMPolicy

class ExtendedAnt():
    def __init__(self, network, custom_env, attack_model, eps):
        self.network = network
        self.custom_env = custom_env
        self.attack_model = attack_model
        self.eps = eps

    def reset(self, uState, attributes=None):
        return self.custom_env.reset(uState, attributes, name="Ant")

    def _get_obs(self):
        qpos = self.env.sim.data.qpos
        qvel = self.env.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def step(self, action, change_filter=False, time_step=None):
        return self.custom_env.step(action[0], change_filter, name="Ant", time_step=time_step)

    def predict(self, state):
        stateTensor = torch.tensor(state).type(torch.FloatTensor).unsqueeze(0)
        action_pds = self.network(stateTensor)
        return torch.clamp(action_pds[0], min=-1, max=1).detach().numpy()

    def opt_attack(self, obs):
        adv_perturbation_pds = self.attack_model(torch.tensor([obs]).type(torch.FloatTensor))
        next_adv_perturbations = self.attack_model.sample(adv_perturbation_pds)
        adv_obs = obs + torch.tanh(next_adv_perturbations).numpy()[0] * self.eps
        return adv_obs

class Ant:
    def __init__(self, model="PPO"):
        # State bounds
        self.bounds = [[-5, 5]]*111

        # Define and load model and optimal attack models
        self.model = CtsPolicy(111, 8, "orthogonal")
        self.attack_model = CtsPolicy(111, 111, "orthogonal")
        if model == "PPO":
            self.checkpoint = torch.load("envs/Ant/Ant_PPO.model")
            self.attack_checkpoint = torch.load("envs/Ant/Ant_Attack_PPO.model")
        elif model == "ATLA":
            self.checkpoint = torch.load("envs/Ant/Ant_ATLA.model")
            self.attack_checkpoint = torch.load("envs/Ant/Ant_Attack_ATLA.model")
        elif model == "LSTM":
            self.checkpoint = torch.load("envs/Ant/Ant_LSTM.model")
            self.attack_checkpoint = torch.load("envs/Ant/Ant_Attack_LSTM.model")
            self.model = CtsLSTMPolicy(111, 8, "orthogonal")
            self.attack_model = CtsLSTMPolicy(111, 111, "orthogonal")
        else:
            exit("Enter valid model choice(PPO, ATLA, or LSTM)")
            
        # Load Model
        self.model.load_state_dict(self.checkpoint['policy_model'])
        self.model.log_stdev.data[:] = -100
        self.model.eval()

        # Load Attack Model
        self.attack_model.load_state_dict(self.attack_checkpoint['adversary_policy_model'])
        self.attack_model.log_stdev.data[:] = -100
        self.attack_model.eval()

        # Noise Level and Time Step
        self.eps = 0.15
        self.period = 0.01

        # Parameters for Zero-One Attack
        self.params = {
            "start_state":np.array([0,0,0.75,1] + [0]*109),
            "eps": 0.15,
            "num_iter":2,
            "step_size":0.15,
            "time_horizon_PPO":20,
            "time_horizon_ATLA":20,
            "time_horizon_LSTM":20,
            "maxlen":1000,
        }
        
        # Load Environment
        self.custom_env = self.checkpoint['envs'][0]
        self.env = ExtendedAnt(self.model, self.custom_env, self.attack_model, self.eps)

        # Miscellaneous 
        self.actionBounds = [[-1, 1]]*8
        self.usesCustom = True
        self.name = "Ant"


        
