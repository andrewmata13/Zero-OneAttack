'''
Terminal value function for the zeroth order search.

The segment objective only sees time_horizon steps of reward. On Half Cheetah the
outcome is decided in the first ~50 steps, where the difference between beginning
to reverse and beginning to accelerate forward is nearly invisible inside a
20-step window - the entire payoff lands beyond the horizon.

V(s) estimates the reward still to come from a terminal state under continued
attack. Adding it to the segment's accumulated reward turns the objective into an
estimate of the final episode reward, which is the quantity actually being
minimized. Because both terms are in true-reward units, no weighting is needed.
'''

import numpy as np
import torch

from torch import nn


class ValueNet(nn.Module):
    'Observation -> reward still to come under continued attack'

    def __init__(self, obs_dim, hidden=64, scale=1000.0):
        super().__init__()
        # Targets are episode-scale returns; predicting in units of `scale` keeps
        # the regression well conditioned
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.trained = False

    def forward(self, obs):
        return self.net(obs).squeeze(-1) * self.scale

    def predict(self, obs):
        'Scalar estimate for one observation; zero until the net has seen data'
        if not self.trained:
            return 0.0
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs)).float().unsqueeze(0)
            return float(self.forward(x)[0])


class ValueBuffer:
    'Stores (observation at a segment boundary, realized reward to go)'

    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.obs = []
        self.togo = []

    def add(self, observations, to_go):
        for o, g in zip(observations, to_go):
            self.obs.append(np.asarray(o, dtype=np.float32))
            self.togo.append(np.float32(g))
        if len(self.obs) > self.capacity:
            self.obs = self.obs[-self.capacity:]
            self.togo = self.togo[-self.capacity:]

    def __len__(self):
        return len(self.obs)


def train_value(net, buffer, epochs=60, batch_size=64, lr=1e-3):
    'Fit V to the realized returns collected so far'

    if len(buffer) < 16:
        return None

    x = torch.tensor(np.array(buffer.obs))
    y = torch.tensor(np.array(buffer.togo))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    net.train()

    last = None
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i+batch_size]
            loss = loss_fn(net(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        last = total / len(x)

    net.eval()
    net.trained = True
    # Report RMSE in reward units, which is easier to sanity check than MSE
    return float(np.sqrt(last))
