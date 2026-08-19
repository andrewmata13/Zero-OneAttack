'''
Student network for warm-starting the zeroth order search.

The student maps a (clean) observation to the target action the attacker wants the
victim policy to emit at that step. Rolling it forward over the look-ahead horizon
produces an initial target action sequence, which is handed to zoopt as an initial
sample. A small refinement budget then improves on it, and the refined sequence is
appended to the replay buffer as fresh on-distribution training data.

Data collection is therefore a byproduct of attacking rather than a prerequisite.
'''

import numpy as np
import torch

from torch import nn


class TargetStudent(nn.Module):
    'Maps an observation to a target action in [-1, 1]'

    def __init__(self, obs_dim, num_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, num_actions), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)

    def predict(self, obs):
        'Target action for a single observation, as a numpy array'
        with torch.no_grad():
            x = torch.tensor(np.asarray(obs)).float().unsqueeze(0)
            return self.forward(x).numpy()[0]


class ReplayBuffer:
    'Fixed-capacity store of (clean observation, refined target action) pairs'

    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.obs = []
        self.targets = []

    def add(self, observations, targets):
        for o, t in zip(observations, targets):
            self.obs.append(np.asarray(o, dtype=np.float32))
            self.targets.append(np.asarray(t, dtype=np.float32))

        # Drop oldest pairs once full so the buffer tracks the current victim
        if len(self.obs) > self.capacity:
            self.obs = self.obs[-self.capacity:]
            self.targets = self.targets[-self.capacity:]

    def __len__(self):
        return len(self.obs)

    def tensors(self):
        return (torch.tensor(np.array(self.obs)),
                torch.tensor(np.array(self.targets)))


def train_student(student, buffer, epochs=20, batch_size=256, lr=1e-3, verbose=False):
    'Fit the student to the refined targets collected so far'

    if len(buffer) < batch_size:
        return None

    x, y = buffer.tensors()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    student.train()

    last = None
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = perm[i:i+batch_size]
            loss = loss_fn(student(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        last = total / len(x)
        if verbose:
            print("  student loss %.5f" % last)

    student.eval()
    return last
