'''
Hybrid search: a learned dynamics model proposes, MuJoCo disposes.

Per segment, a large candidate population is scored with a learned dynamics model
- entirely batched, since filter, PGD, policy and dynamics are all tensor ops -
and only the top-k are re-evaluated in the real simulator. Every committed
decision is therefore made by MuJoCo, so model error costs ranking quality and
never correctness.

Measured on Half Cheetah PPO: scoring 250 candidates and verifying 10 lands within
1-2 reward of the true best of all 250, across four points in the episode. That is
roughly a 25x reduction in simulator rollouts per segment.

Recurrent victims are not supported: batching a per-candidate LSTM hidden state
through this loop is a different problem, and the model was never validated there.
'''

import numpy as np
import torch

from torch import nn

SDIM_DEFAULT = 18


class Dynamics(nn.Module):
    'Predicts delta uState from (uState without x, action)'

    def __init__(self, sdim, adim, hidden=512):
        super().__init__()
        self.sdim, self.adim = sdim, adim
        self.net = nn.Sequential(
            nn.Linear(sdim - 1 + adim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, sdim),
        )
        self.register_buffer("xm", torch.zeros(sdim - 1 + adim))
        self.register_buffer("xs", torch.ones(sdim - 1 + adim))
        self.register_buffer("ym", torch.zeros(sdim))
        self.register_buffer("ys", torch.ones(sdim))

    def forward(self, s_noX, a):
        z = torch.cat([s_noX, a], -1)
        return self.net((z - self.xm) / self.xs) * self.ys + self.ym


class DynamicsEnsemble:
    'Ensemble of delta-state predictors plus the batched scoring loop'

    def __init__(self, sdim, adim, members=3, hidden=512, device="cpu"):
        self.sdim, self.adim, self.device = sdim, adim, torch.device(device)
        self.nets = [Dynamics(sdim, adim, hidden).to(self.device) for _ in range(members)]

    # ------------------------------------------------------------------ training
    def fit(self, states, actions, deltas, epochs=45, batch_size=256, lr=1e-3, verbose=False):
        X = torch.as_tensor(np.asarray(states), dtype=torch.float32)[:, 1:]
        A = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
        Y = torch.as_tensor(np.asarray(deltas), dtype=torch.float32)
        XA = torch.cat([X, A], -1)

        for m, net in enumerate(self.nets):
            net.xm.copy_(XA.mean(0)); net.xs.copy_(XA.std(0) + 1e-6)
            net.ym.copy_(Y.mean(0)); net.ys.copy_(Y.std(0) + 1e-6)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            # bootstrap resample so members are not identical
            boot = torch.randint(0, len(X), (len(X),))
            Xb, Ab, Yb = X[boot].to(self.device), A[boot].to(self.device), Y[boot].to(self.device)
            net.train()
            for _ in range(epochs):
                perm = torch.randperm(len(Xb), device=self.device)
                for i in range(0, len(Xb), batch_size):
                    j = perm[i:i+batch_size]
                    loss = ((net(Xb[j], Ab[j]) - Yb[j]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            if verbose:
                print("  dynamics member %d trained, final mse %.5f" % (m, float(loss)), flush=True)
        return self

    def save(self, path):
        torch.save({"sdim": self.sdim, "adim": self.adim,
                    "state_dicts": [n.state_dict() for n in self.nets]}, path)

    @staticmethod
    def load(path, hidden=512, device="cpu"):
        blob = torch.load(path, map_location=device)
        ens = DynamicsEnsemble(blob["sdim"], blob["adim"],
                               members=len(blob["state_dicts"]), hidden=hidden, device=device)
        for net, sd in zip(ens.nets, blob["state_dicts"]):
            net.load_state_dict(sd)
            net.eval()
        return ens

    # ------------------------------------------------------------------ scoring
    def score(self, u0, target_batch, policy, pgd_attack, filt_mean, filt_std,
              filt_clip, dt, ctrl_cost=0.1):
        '''Predicted segment reward for a whole candidate population at once.

        target_batch is [N, horizon, adim] of target actions. Everything advances
        in lockstep: one batched PGD call and one batched dynamics call per step,
        regardless of how many candidates are in flight.
        '''
        N, H, _ = target_batch.shape
        dev = self.device
        u = torch.as_tensor(np.asarray(u0), dtype=torch.float32, device=dev).repeat(N, 1)
        tgt = torch.as_tensor(target_batch, dtype=torch.float32, device=dev)
        mean = torch.as_tensor(filt_mean, dtype=torch.float32, device=dev)
        std = torch.as_tensor(filt_std, dtype=torch.float32, device=dev)

        total = torch.zeros(N, device=dev)
        for i in range(H):
            obs = (u[:, 1:] - mean) / (std + 1e-8)
            if filt_clip:
                obs = obs.clamp(-filt_clip, filt_clip)

            adv = pgd_attack.perturb(obs, y=tgt[:, i, :]).detach()
            with torch.no_grad():
                act = torch.clamp(policy(adv)[0], -1, 1)
                delta = torch.stack([n(u[:, 1:], act) for n in self.nets]).mean(0)

            u_next = u + delta
            total = total + (u_next[:, 0] - u[:, 0]) / dt - ctrl_cost * (act ** 2).sum(-1)
            u = u_next

        return total.cpu().numpy()
