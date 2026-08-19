import os
import sys
import time
import pickle
import random
import torch
import zoopt
import numpy as np

from matplotlib import pyplot as plt
from numpy.linalg import norm

from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from copy import deepcopy

from advertorch.attacks import PGDAttack, LinfPGDAttack, L2PGDAttack

from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt, Solution
from zoopt.utils.tool_function import ToolFunction

from util import AdvertorchAdapter, AdversarialTrajectory, calculateError, round_obs
from timeutils import Timers

class OptimizationAttack:
    def __init__(self, environment=None, start_state=None, eps=None, num_iter=15, step_size=0.01, time_horizon=20, maxlen=1000, net_type=None, attack_type="state", attack_budget=None, basis=None, basis_type="block", optimizer="racos", legacy_pgd=False, warm_start=None, commit=None, abandon_margin=None, verbose=False, pgd_mode=None, target_space="continuous", cem_pop=100, cem_elite=0.15):

        # Load Environment
        self.env_constructor = environment
        self.net_type = net_type
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

        # Reset Hidden State (if needed)
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
        self.attack_budget = attack_budget

        # Initialize Adversarial Trajectories
        self.adv_trajs = []

        # Attack Types: "state", "time", "state+time"
        self.attack_type = attack_type

        #################################################################################
        # Optional low-dimensional parameterization of the target action sequence.      #
        # basis=None searches one free variable per step (dim = num_actions*horizon).   #
        # basis=k searches k coefficients per action instead, so the search dimension   #
        # is decoupled from the look-ahead horizon. Only used for "state" attacks.      #
        #################################################################################
        self.basis = basis
        self.basis_type = basis_type

        #################################################################################
        # Target space. "continuous" searches targets anywhere in [-1,1], whose mean    #
        # magnitude is only 0.5, so most candidates ask the victim for a moderate       #
        # action. "corners" searches only the saturated targets, which is what actually #
        # breaks a gait: measured, CrossEntropy reached saturated actions 35.8% of the  #
        # time against 20.2% for a faithful targeted attack.                            #
        #################################################################################
        self.target_space = target_space

        # Zeroth order solver: "racos" (zoopt) or "random" (uniform sampling control)
        self.optimizer = optimizer

        ##################################################################################
        # The archived Adv_Traj runs used advertorch's defaults, which are wrong here:   #
        #   - loss_fn defaults to CrossEntropyLoss over the continuous action vector     #
        #   - targeted defaults to False, so PGD ascends AWAY from the target action     #
        #   - rand_init defaults to True, making the cost function stochastic (measured  #
        #     noise/signal 0.79, which the zeroth order search cannot see through)       #
        # legacy_pgd=True reproduces that behaviour; the default is the intended         #
        # targeted attack evaluated on a deterministic objective.                        #
        ##################################################################################
        self.legacy_pgd = legacy_pgd
        self.pgd_mode = pgd_mode or ("legacy" if legacy_pgd else "fixed")

        #############################################################################
        # Optional student used to warm-start the zeroth order search. Set via      #
        # attach_student(); every segment it warm-starts also emits the refined     #
        # target sequence back into the buffer as training data.                    #
        #############################################################################
        self.student = None
        self.student_buffer = None
        self.best_clean_obs = None
        self.best_targets = None

        #############################################################################
        # MPC-style warm start: seed each segment with the previous segment's plan  #
        # shifted forward by the number of steps that were committed. With          #
        # commit == time_horizon the whole plan has been consumed, so the shift     #
        # wraps around and this degenerates to reusing the previous plan, which is  #
        # still a reasonable prior for the periodic gaits these victims produce.    #
        #############################################################################
        self.warm_start = warm_start
        self.commit = commit
        self.prev_targets = None

        #############################################################################
        # Early abandonment: stop simulating a candidate once its partial reward is #
        # already worse than the incumbent's at the same step by more than the      #
        # margin. Reward here is dense, so most candidates are visibly bad within a #
        # few steps of the horizon. None disables it.                               #
        #############################################################################
        self.abandon_margin = abandon_margin
        self.best_profile = None
        self.abandoned = 0
        self.evaluated = 0

        #############################################################################
        # Terminal value function. The segment objective only sees time_horizon     #
        # steps, but on Cheetah the outcome is decided in the opening segments      #
        # where the payoff lands entirely beyond that window. Adding V(terminal)    #
        # makes the objective an estimate of the final episode reward. Attach with  #
        # attach_value(); both terms are in true-reward units so no weight is       #
        # needed.                                                                   #
        #############################################################################
        self.value_net = None
        self.value_buffer = None
        self.segment_obs = []
        self.segment_reward = []

        #############################################################################
        # Learned dynamics model used to shortlist candidates (see hybrid.py).      #
        # Attach with attach_dynamics() and set optimizer="hybrid". Real rollouts   #
        # can also record transitions, so the attack generates its own training     #
        # data for free.                                                            #
        #############################################################################
        self.dynamics = None
        self.cem_pop = cem_pop
        self.cem_elite = cem_elite
        self.n_propose = 2000
        self.n_verify = 10
        self.transitions = []
        self.collect_transitions = False

        # Per-step debug printing, off by default (these prints are in the hot loop)
        self.verbose = verbose
        if not verbose:
            # zoopt logs every candidate evaluation, which dominates the output and
            # costs real time inside the search
            ToolFunction.log = staticmethod(lambda text: None)

        # Constants for PGD Attack
        self.MIN_BOUND = -1000
        self.MAX_BOUND = 1000

    def build_pgd(self):
        """Construct the PGD inner solve.

        pgd_mode separates the three things the "correctness fix" bundled together:
          legacy        CrossEntropy, untargeted, random init  (what the archive used)
          fixed         MSE, targeted, deterministic
          legacy_norand CrossEntropy, untargeted, deterministic
          fixed_rand    MSE, targeted, random init
        Randomised init is not only noise: the zeroth order search covers at most
        ~120 dimensions of target space, while PGD's random start explores the much
        larger perturbation space, so the same target yields different adversarial
        observations on repeat evaluations.
        """
        mode = self.pgd_mode
        targeted = mode in ("fixed", "fixed_rand")
        rand_init = mode in ("legacy", "fixed_rand")
        loss_fn = nn.MSELoss(reduction="sum") if targeted else None

        return PGDAttack(self.pgdModel, loss_fn=loss_fn, eps=self.eps,
                         nb_iter=self.num_iter, eps_iter=self.step_size,
                         clip_min=self.MIN_BOUND, clip_max=self.MAX_BOUND,
                         targeted=targeted, rand_init=rand_init)

    def attach_student(self, student, buffer):
        'Use a student to warm-start the search and collect its refinements'
        self.student = student
        self.student_buffer = buffer

    def attach_dynamics(self, ensemble, n_propose=2000, n_verify=10):
        '''Use a learned dynamics model to shortlist candidates. Only n_verify of
        them are ever simulated, so MuJoCo still makes every committed decision.'''
        self.dynamics = ensemble
        self.n_propose = n_propose
        self.n_verify = n_verify

    def filter_stats(self):
        'Frozen mean/std/clip of the observation filter, as plain arrays'
        f = self.env.custom_env.new_filter
        while not hasattr(f, "rs") and hasattr(f, "prev_filter"):
            f = f.prev_filter
        return np.array(f.rs.mean), np.array(f.rs.std), getattr(f, "clip", None)

    def attach_value(self, value_net, buffer):
        'Score each candidate by segment reward plus predicted reward to come'
        self.value_net = value_net
        self.value_buffer = buffer

    def sarl_targets(self, current_state, env_attr):
        '''Roll Zhang's trained adversary forward and record the action the victim
        actually emits under its perturbation. That realized action sequence is a
        strong, already-trained prior for the zeroth order search - and unlike a
        regression student it is a specific sequence, not an average of many.'''

        obs = self.env.reset(current_state, env_attr)

        if self.net_type == "LSTM":
            self.env.network.hidden = deepcopy(self.current_hidden)

        targets = []
        for _ in range(self.time_horizon):
            adv_obs = self.env.opt_attack(obs)
            action = self.env.predict(adv_obs)
            targets.append(np.array(action).flatten()[:self.num_actions])

            state, _, done, _ = self.env.step(action)
            if done:
                while len(targets) < self.time_horizon:
                    targets.append(targets[-1])
                break
            obs = state[1]

        return np.array(targets)

    def project_to_basis(self, targets):
        '''Project a per-step target action sequence (horizon x num_actions) onto the
        optimizer variables, so a student rollout can be handed to zoopt as an
        initial sample'''

        if self.basis is None:
            return list(np.asarray(targets).flatten())

        targets = np.asarray(targets)
        if self.basis_type == "block":
            # Block coefficient is the mean target over the steps in that block
            idx = (np.arange(self.time_horizon) * self.basis) // self.time_horizon
            coeffs = np.array([targets[idx == b].mean(axis=0) for b in range(self.basis)])
        else:
            # Least squares fit of the cosine series to the student's sequence
            t = (np.arange(self.time_horizon) + 0.5) / self.time_horizon
            cosines = np.cos(np.pi * np.outer(t, np.arange(self.basis)))
            coeffs = np.linalg.lstsq(cosines, targets, rcond=None)[0]

        return list(np.clip(coeffs, -1, 1).flatten())

    def action_dim_size(self):
        # Number of optimizer variables spent on target actions
        if self.basis is None:
            return self.num_actions * self.time_horizon

        return self.num_actions * self.basis

    def expand_actions(self, coeffs):
        '''Decode optimizer variables into a flat target action sequence of
        length num_actions * time_horizon'''

        coeffs = np.asarray(coeffs, dtype=float)
        if self.target_space == "corners":
            coeffs = 2.0 * coeffs - 1.0

        if self.basis is None:
            return coeffs

        # Coefficients are laid out as (basis index, action index)
        c = coeffs.reshape(self.basis, self.num_actions)

        if self.basis_type == "block":
            # Piecewise-constant: each block holds one target action for horizon/basis steps
            idx = (np.arange(self.time_horizon) * self.basis) // self.time_horizon
            seq = c[idx]
        elif self.basis_type == "dct":
            # Smooth: low-order cosine series, clipped back to the action bounds
            t = (np.arange(self.time_horizon) + 0.5) / self.time_horizon
            cosines = np.cos(np.pi * np.outer(t, np.arange(self.basis)))
            seq = np.clip(cosines.dot(c), -1, 1)
        else:
            exit("Invalid basis type specified")

        return seq.flatten()

    def optimize(self, current_state, env_attr):
        # Keep Track of Best Actions/Rewards Found in Optimizer
        self.best_obs = None
        self.best_reward = 1e10
        self.best_action = None
        self.done = False
        self.best_delay = None
        self.best_profile = None

        # Reset Environment
        obs = self.env.reset(current_state, env_attr)

        # Initialize PGD Attack (stateless apart from the model reference, so it is
        # built once per optimize() call rather than once per candidate rollout)
        attack = self.build_pgd()

        # First-order Cost Function with Gradient-Based Attack
        def cost_function(u):
            # Extract input from zoopt wrapper
            if isinstance(u, zoopt.solution.Solution):
                u = u.get_x()

            # Reset Environment
            obs = self.env.reset(current_state, env_attr)

            # Reset Model Hidden State to Beginning Hidden State
            if self.net_type == "LSTM":
                self.env.network.hidden = deepcopy(self.current_hidden)
            
            # Separate timing params from action params
            opt_actions = []
            timing = []
            if self.attack_type == "state+time":
                for i, param in enumerate(u):
                    if i%(self.num_actions + 1) == 0:
                        timing.append(param)
                    else:
                        opt_actions.append(param)
            elif self.attack_type == "state":
                opt_actions = self.expand_actions(u)
            else:
                timing = u

            #######################################################################
            # Run environment loop with control commands augmented with pgd attack#    
            #######################################################################
            
            current_time = 0
            total_reward = 0
            observations = []
            actions = []
            clean_observations = []
            profile = []
            prev_u = None
            if self.collect_transitions:
                # Full MuJoCo state, matching what env.step returns as state[0]
                prev_u = np.array(self.env._get_obs())
            self.evaluated += 1
            for i in range(self.time_horizon):
                # For LSTM networks, set hidden state for PGD Model to Network hidden state
                if self.net_type == "LSTM":
                    self.pgdModel.net.hidden = deepcopy([self.env.network.hidden[0].detach(), self.env.network.hidden[1].detach()])

                # Keep the unperturbed observation so the student can be trained on
                # (clean observation -> refined target action) pairs
                clean_observations.append(np.array(obs))

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

                obs = state[1]
                total_reward += reward

                ###################################################################
                # Early abandonment. The returned value is always worse than the  #
                # incumbent, but stays ordered by how bad the prefix was so the   #
                # optimizer still gets usable ranking information.                #
                ###################################################################
                if self.collect_transitions:
                    u_next = np.array(state[0])
                    self.transitions.append((prev_u.copy(),
                                             np.array(adv_action).flatten()[:self.num_actions],
                                             u_next - prev_u))
                    prev_u = u_next

                profile.append(self.env.custom_env.total_true_reward)
                if (self.abandon_margin is not None and self.best_profile is not None
                        and i < len(self.best_profile)
                        and profile[i] > self.best_profile[i] + self.abandon_margin):
                    self.abandoned += 1
                    self.done = False
                    return self.best_reward + (profile[i] - self.best_profile[i])

            # Get total reward (only in Zhang's custom envs)
            total_reward = self.env.custom_env.total_true_reward

            ###################################################################
            # Terminal value: reward accumulated so far plus reward still to  #
            # come is an estimate of the final episode reward, which is what  #
            # the attack actually minimizes. Both terms are true-reward units.#
            ###################################################################
            if self.value_net is not None and not self.done and len(clean_observations):
                total_reward = total_reward + self.value_net.predict(obs)

            def record_best():
                # Snapshot the incumbent rollout. best_clean_obs/best_targets are the
                # (observation -> target action) pairs the student is trained on.
                self.best_obs = observations
                self.best_action = actions
                self.best_delay = timing
                self.best_clean_obs = clean_observations
                self.best_profile = profile
                if self.attack_type == "time":
                    self.best_targets = None
                else:
                    n = len(clean_observations)
                    self.best_targets = np.array(opt_actions[:n*self.num_actions]).reshape(n, self.num_actions)

            # Keep Track of Best Found Reward
            if total_reward < self.best_reward and not self.done:
                if self.name == "Ant" or self.name == "Half Cheetah":
                    ########################################################################################################
                    # For non-safety critical envs, simulate forward to see if the environment will reach a "bad" state    # 
                    # (done condition = True) and try to avoid those, will allow us to find the backwards-moving behaviors)#
                    ########################################################################################################

                    flag = False
                    for i in range(self.time_horizon):
                        action = self.env.predict(obs)
                        state, reward, done, _ = self.env.step(action)
                        obs = state[1]
                        if done:
                            flag = True
                    if not flag or self.best_action is None:
                        self.best_reward = total_reward
                        record_best()
                            
                elif self.name == "Hopper" or self.name == "Walker2D":
                    # Not needed for safety-critical ones, we want to find an unsafe trace asap
                    if self.best_action is None:
                        self.best_reward = total_reward
                        record_best()
            elif total_reward < self.best_reward and self.done:
                if self.name == "Hopper" or self.name == "Walker2D":
                    self.best_reward = total_reward
                    record_best()
                    ##################################################################################
                    # Add additional reward constant for the optimizer for unsafe traces to encourage#
                    # finding unsafe traces for the safety critical envs, slight reward tuning       #
                    ##################################################################################
                    total_reward -= 1000
                elif self.name == "Ant" or self.name == "Half Cheetah":
                    if self.best_action is None:
                        record_best()
                    
            self.done = False
            return total_reward

        # Set-up Zeroth Order Optimization For Each Attack Type
        Timers.tic("Global Optimization")
        if self.attack_type == "state":
            dim_size = self.action_dim_size()
            if self.target_space == "corners":
                # binary variables, decoded to -1/+1 in expand_actions
                bounds = [[0, 1]]*dim_size
                dim = Dimension(dim_size, bounds, [False]*dim_size)
            else:
                bounds = [[-1, 1]]*dim_size
                dim = Dimension(dim_size, bounds, [True]*dim_size)
        elif self.attack_type == "time":
            dim_size = self.time_horizon
            bounds = []
            for i in range(dim_size):
                bounds.append([-self.period/2, self.period/2])

            dim = Dimension(dim_size, bounds, [True]*dim_size)
        elif self.attack_type == "state+time":
            dim_size = (self.num_actions + 1) * self.time_horizon
            bounds = []
            for i in range(dim_size):
                if i % (self.num_actions + 1) == 0:
                    bounds.append([-self.period/2, self.period/2])
                else:
                    bounds.append([-1,1])
            dim = Dimension(dim_size, bounds, [True]*dim_size)
        else:
            exit("Invalid attack type specified")
        
        ###################################################################
        # Warm start is computed once, before the solver dispatch, so every
        # search mode can seed from it
        ###################################################################
        init_warm = None
        if self.attack_type == "state":
            warm = None
            if self.warm_start == "sarl":
                # Zhang's trained adversary as the search prior
                warm = self.sarl_targets(current_state, env_attr)
            elif self.student is not None:
                warm = self.student_targets(current_state, env_attr)
            elif self.warm_start == "shift" and self.prev_targets is not None:
                shift = (self.commit or self.time_horizon) % self.time_horizon
                warm = np.roll(self.prev_targets, -shift, axis=0)
            if warm is not None:
                init_warm = self.project_to_basis(warm)

        if self.optimizer == "hybrid":
            #######################################################################
            # Propose a large population, score it with the learned dynamics      #
            # model in one batched pass, then spend the real simulator budget     #
            # only on the top n_verify. cost_function tracks the incumbent, so    #
            # the committed plan is always one MuJoCo actually evaluated.         #
            #######################################################################
            if self.dynamics is None:
                exit("optimizer='hybrid' requires attach_dynamics()")
            if self.net_type == "LSTM":
                exit("hybrid search does not support recurrent victims")

            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            pop = np.random.uniform(lo, hi, size=(self.n_propose, len(bounds)))

            # Keep the warm start in the population rather than discarding it
            if init_warm is not None:
                pop[0] = np.clip(init_warm, lo, hi)

            targets = np.stack([np.asarray(self.expand_actions(c)).reshape(
                self.time_horizon, self.num_actions) for c in pop])

            # ExtendedEnv._get_obs returns concat(qpos, qvel), which is uState
            u0 = np.array(self.env._get_obs())
            mean, std, clip = self.filter_stats()

            Timers.tic("Model Scoring")
            scores = self.dynamics.score(
                u0, targets, self.environment.model, attack, mean, std, clip,
                dt=self.env.custom_env.env.dt)
            Timers.toc("Model Scoring")

            for idx in np.argsort(scores)[:self.n_verify]:
                cost_function(list(pop[idx]))

        elif self.optimizer == "random":
            #######################################################################
            # Control: spend the same budget on uniform sampling. cost_function   #
            # already tracks the incumbent, so nothing else is needed here. Used  #
            # to measure how much the model-based search actually contributes.    #
            #######################################################################
            for _ in range(self.attack_budget):
                cost_function([np.random.uniform(lo, hi) for lo, hi in bounds])

        elif self.optimizer == "cem":
            #######################################################################
            # Cross-Entropy Method: a batchable adaptive search. Each generation  #
            # samples a population from a Gaussian, keeps the elite fraction (the  #
            # lowest rewards, since we minimize), and refits the Gaussian to them. #
            # Candidates within a generation are independent, so this is what a    #
            # GPU/MJX port would batch. Here it runs serially to test STRENGTH     #
            # only: does adaptive population search match SRACOS at equal budget?  #
            #######################################################################
            lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
            pop_size = self.cem_pop
            n_gen = max(1, self.attack_budget // pop_size)
            n_elite = max(2, int(round(pop_size * self.cem_elite)))

            mean = np.array(init_warm, dtype=float) if init_warm is not None \
                   else (lo + hi) / 2.0
            std = (hi - lo) / 4.0            # ~covers the box at +-2 std

            for g in range(n_gen):
                samples = np.clip(mean + std * np.random.randn(pop_size, len(bounds)), lo, hi)
                rewards = np.array([cost_function(list(x)) for x in samples])
                elite = samples[np.argsort(rewards)[:n_elite]]
                mean = elite.mean(axis=0)
                # floor the std so the distribution does not collapse prematurely
                std = np.maximum(elite.std(axis=0), 0.02 * (hi - lo))
        else:
            obj = Objective(cost_function, dim)

            init_samples = [Solution(x=init_warm)] if init_warm is not None else None

            solution = Opt.min(obj, Parameter(budget=self.attack_budget, init_samples=init_samples))
        Timers.toc("Global Optimization")

        # Refined targets go back to the buffer as on-distribution training data
        if self.student_buffer is not None and self.best_targets is not None:
            self.student_buffer.add(self.best_clean_obs, self.best_targets)

        # Carry the plan to the next segment, padding if the rollout ended early
        if self.best_targets is not None:
            plan = self.best_targets
            if len(plan) < self.time_horizon:
                pad = np.repeat(plan[-1:], self.time_horizon - len(plan), axis=0)
                plan = np.concatenate([plan, pad])
            self.prev_targets = plan

        return self.best_action, self.best_obs, self.done, self.best_delay

    def student_targets(self, current_state, env_attr):
        '''Roll the student forward over the look-ahead horizon, applying the PGD
        attack toward each proposed target, and return the target sequence'''

        obs = self.env.reset(current_state, env_attr)

        if self.net_type == "LSTM":
            self.env.network.hidden = deepcopy(self.current_hidden)

        attack = self.build_pgd()

        targets = []
        for _ in range(self.time_horizon):
            if self.net_type == "LSTM":
                self.pgdModel.net.hidden = deepcopy([self.env.network.hidden[0].detach(), self.env.network.hidden[1].detach()])

            target = self.student.predict(obs)
            targets.append(target)

            y = torch.tensor(np.array([target])).type(torch.FloatTensor)
            torch_state = torch.tensor(obs).type(torch.FloatTensor)
            adv_obs = attack.perturb(torch_state.unsqueeze(0), y=y).detach().numpy()[0]

            state, _, done, _ = self.env.step(self.env.predict(adv_obs))
            if done:
                # Pad the remainder so the projection still sees a full horizon
                while len(targets) < self.time_horizon:
                    targets.append(target)
                break
            obs = state[1]

        return np.array(targets)
                     
    def next_adv(self, current_state, env_attr):
        # Get Next Adversarial Actions/States
        actions, adv_states, done, timing = self.optimize(current_state, env_attr)

        # Reset Environment
        obs = self.env.reset(current_state, env_attr)

        # Keep Track of Hidden State (for LSTM)
        if self.net_type == "LSTM":
            self.env.network.hidden = deepcopy(self.current_hidden)

        # Do some sanity checks to make sure we are in the right noise level + advance state
        current_time = 0

        for i in range(len(actions)):
            # Compute Noise to Check within Noise Cap
            adv = adv_states[i]
            noise = calculateError(np.array(obs), adv, self.bounds)
            if self.verbose:
                print(obs, adv)
                print("Noise Found:", noise)

            if self.attack_type == "state":
                # calculateError normalizes by the state bound range, so it cannot be
                # compared to eps directly. PGD constrains the raw L-inf perturbation,
                # so check that instead (tolerance covers float32 round-tripping).
                linf = np.max(np.abs(np.array(obs) - np.array(adv)))
                assert linf <= self.eps + 1e-4, "Perturbation %f exceeds eps %f" % (linf, self.eps)

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
            
            obs = state[1]
        
        if self.verbose:
            print("Total Reward:", self.env.custom_env.total_true_reward)
        
        return state, adv_states, actions, done, timing
        
    def generate(self, num_traj=1, only_index=None):
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

        ###############################################################################
        # Trajectories are independent, so a worker process can attack just one of    #
        # them. The full seeded list is still built first, so the starting state for  #
        # a given index is identical to what a sequential run would have used.        #
        ###############################################################################
        if only_index is not None:
            starting_states = [starting_states[only_index]]

        self.current_time = 0
            
        for state in starting_states:
            # Environment For Confirming Trajectory
            test_env = self.env_constructor(self.net_type).env
            test_env.custom_env.env.seed(num_traj)

            # Plans do not carry across trajectories
            self.prev_targets = None

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
                # Store Hidden State (for LSTM)
                if self.net_type == "LSTM":
                    self.current_hidden = deepcopy([test_env.network.hidden[0].detach(), test_env.network.hidden[1].detach()])
                    
                if self.verbose:
                    print("Attack Iteration:", i)
                
                # Cumulative reward entering this segment, so reward-to-go can be
                # computed for the value function once the trajectory finishes
                seg_start_reward = test_env.custom_env.total_true_reward

                state, adv_states, adv_actions, done, timing = self.next_adv(current_state, env_attr)

                if self.value_buffer is not None and self.best_clean_obs:
                    self.segment_obs.append(np.array(self.best_clean_obs[0]))
                    self.segment_reward.append(seg_start_reward)

                env_attr = state[2:]
                current_state = state[0]

                all_states += list(adv_states)
                all_actions += list(adv_actions)
                all_timing += list(timing)
                
                # Rerun on Confirmation Environment
                current_time = 0

                input_state = adv_states[0]

                for i in range(len(adv_actions)):
                    action = test_env.predict(adv_states[i])
                        
                    if self.attack_type == "state":
                        test_state, _, step_done, _ = test_env.step(action, change_filter=True)
                    else:
                        start_time = (i+1) * self.period
                        if i == self.time_horizon - 1:
                            time_step = start_time - current_time
                        else:
                            time_step = (start_time - current_time) + timing[i]

                        test_state, _, step_done, _ = test_env.step(action, change_filter=True, time_step=time_step)
                        current_time += time_step

                    # cost_function clears self.done before returning, so the flag from
                    # optimize() is always False. The confirmation rollout is the ground
                    # truth for whether the victim actually failed.
                    done = done or step_done

                    input_state = test_state[1]
                               
                if self.verbose:
                    print("REWARD:", test_env.custom_env.total_true_reward)
                    
                self.env = self.env_constructor(self.net_type).env
                self.env.reset(test_state[0], test_state[2:])
                current_state = deepcopy(test_state[0])
                if self.net_type == "LSTM":
                    self.env.network.hidden = deepcopy([test_env.network.hidden[0].detach(), test_env.network.hidden[1].detach()])

                # Done Condition For Walker and Hopper, stop simulation if met
                if done and (self.name == "Walker2D" or self.name == "Hopper"):
                    break

            # Add Adversarial Trajectory
            # Realized reward to go from each segment boundary trains the value net
            if self.value_buffer is not None and self.segment_obs:
                final_reward = test_env.custom_env.total_true_reward
                self.value_buffer.add(self.segment_obs,
                                      [final_reward - r for r in self.segment_reward])
                self.segment_obs, self.segment_reward = [], []

            traj = AdversarialTrajectory(self.start_state, all_states, all_actions, all_timing, self.env.custom_env.total_true_reward)
            self.adv_trajs.append(traj)
                
        #Timers.toc("Top")
        #Timers.print_stats()
            

    '''
    Method to be painfully sure that the trajectory is valid, shouldn't usually be used
    '''
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

            if render:
                confirm_env.custom_env.env.render()
                time.sleep(0.001)

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
                adv_obs = test_env.opt_attack(obs)       

                action = test_env.predict(adv_obs)

                state, reward, done, _ = test_env.step(action, change_filter=True)
                obs = state[1]

                step += 1

                if done and (self.name == "Hopper" or self.name == "Walker2D"):
                    break

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



