'''
Reference implementation of the victim policy + legacy_norand PGD in pure numpy.

This is the CONTRACT the JAX port must reproduce. It runs without JAX (verify on
any box), and jax_port/attack_mjx.py mirrors it with jax.numpy + vmap. The manual
PGD here was verified to match advertorch to 0.00e+00.

Policy: CtsPolicy, obs(17) -> Linear(17,64) tanh -> Linear(64,64) tanh ->
        final_mean Linear(64,6). AdvertorchAdapter uses the raw mean (no clamp);
        the ENV clamps the executed action to [-1,1].
'''
import numpy as np
import torch


def load_weights(model_path):
    'Pull the CtsPolicy weights out of a victim checkpoint as numpy arrays.'
    ckpt = torch.load(model_path, map_location="cpu")
    sd = ckpt["policy_model"]
    g = lambda k: sd[k].cpu().numpy()
    return dict(
        W0=g("affine_layers.0.weight"), b0=g("affine_layers.0.bias"),
        W1=g("affine_layers.1.weight"), b1=g("affine_layers.1.bias"),
        Wm=g("final_mean.weight"),      bm=g("final_mean.bias"),
    )


def policy_mean(w, x):
    'Raw policy mean action (what PGD attacks). x: [N,17] -> [N,6]. numpy.'
    h = np.tanh(x @ w["W0"].T + w["b0"])
    h = np.tanh(h @ w["W1"].T + w["b1"])
    return h @ w["Wm"].T + w["bm"]


def log_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def legacy_pgd(w, x, y, eps, nb_iter=1, eps_iter=None):
    '''Reproduce legacy_norand PGD: CrossEntropy(soft target), untargeted (ascend),
    deterministic init. x:[N,17] obs, y:[N,6] proposed target actions -> adv obs.
    Batched; this is the per-step map the search calls for the whole population.'''
    eps_iter = eps_iter if eps_iter is not None else eps
    delta = np.zeros_like(x)
    for _ in range(nb_iter):
        # gradient of  -sum(y * log_softmax(policy_mean(x+delta)))  wrt delta
        xin = x + delta
        h0 = np.tanh(xin @ w["W0"].T + w["b0"])
        h1 = np.tanh(h0 @ w["W1"].T + w["b1"])
        logits = h1 @ w["Wm"].T + w["bm"]
        p = np.exp(log_softmax(logits))
        # d loss/d logits = softmax(logits)*sum(y) - y   (for -sum(y*logsoftmax))
        dlogits = p * y.sum(axis=-1, keepdims=True) - y
        dh1 = dlogits @ w["Wm"] * (1 - h1**2)
        dh0 = dh1 @ w["W1"] * (1 - h0**2)
        dx = dh0 @ w["W0"]
        delta = np.clip(delta + eps_iter * np.sign(dx), -eps, eps)
    return np.clip(x + delta, -1000, 1000)


if __name__ == "__main__":
    # self-check against the live torch policy + advertorch, if available
    import os, sys, warnings; warnings.filterwarnings("ignore")
    HERE=os.path.dirname(os.path.abspath(__file__)); OPT=os.path.dirname(HERE)
    sys.path.insert(0, OPT); os.chdir(OPT)
    from envs.HalfCheetah.CheetahEnv import HalfCheetah
    from util import AdvertorchAdapter
    from advertorch.attacks import PGDAttack
    h = HalfCheetah("PPO"); w = load_weights("envs/HalfCheetah/HalfCheetah_PPO.model")
    net = AdvertorchAdapter(h.model); net.eval()
    obs = np.array(h.env.reset(h.params["start_state"], None))
    x = np.tile(obs, (8, 1)); 
    # policy match
    tm = net(torch.tensor(x).float()).detach().numpy()
    print("policy mean max-diff:", np.abs(policy_mean(w, x) - tm).max())
    # pgd match
    y = np.random.uniform(-1, 1, (8, 6))
    atk = PGDAttack(net, eps=0.15, nb_iter=1, eps_iter=0.15, clip_min=-1000, clip_max=1000, rand_init=False)
    ta = atk.perturb(torch.tensor(x).float(), y=torch.tensor(y).float()).detach().numpy()
    print("legacy PGD max-diff:", np.abs(legacy_pgd(w, x, y, 0.15) - ta).max())
