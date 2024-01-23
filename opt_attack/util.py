from torch import nn
from torch.nn import functional as F
import numpy as np

# Adapter Class for PGD Attacks
class AdvertorchAdapter(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, obs, last_state=None):
        action_pds = self.net(obs)
        return action_pds[0]

    
class AdversarialTrajectory:
    def __init__(self, start_state, adv_states, adv_actions, timing, reward):
        self.start_state = start_state
        self.adv_states = adv_states
        self.adv_actions = adv_actions
        self.timing = timing
        self.reward = reward

        
# Calculate L-inf Percent Error
def calculateError(clean, noisy, bounds):
    errs = []
    for i in range(len(clean)):
        errs.append(abs(clean[i] - noisy[i]) / (bounds[i][1] - bounds[i][0] + 1e-8))
    return max(errs)


def round_obs(clean, noisy, eps):
    max_array = np.array(clean) + eps
    min_array = np.array(clean) - eps
    new_obs = []
    for i, item in enumerate(noisy):
        if item > max_array[i]:
            new_obs.append(max_array[i] - 1e-10)
        elif item < min_array[i]:
            new_obs.append(min_array[i] + 1e-10)
        else:
            new_obs.append(item)

    return np.array(new_obs)
