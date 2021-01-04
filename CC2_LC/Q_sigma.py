import numpy as np
import scipy
import random

from Agent import LearningAgent
from math_tools import argMax
from CountState import CountState


class Q_sigma(LearningAgent):
    def __init__(self, solution_params, problem_params, state_rep, type_rep):
        super().__init__(solution_params, problem_params, state_rep, type_rep)
        self.sigma_label = solution_params["sigma"]
        # print("label", self.sigma_label)
        if isinstance(self.sigma_label, str):
            self.sigma_0 = 1.0
        else:
            self.sigma_0 = self.sigma_label

    def get_sigma(self, t, t_ep, count_s):
       # if self.sigma_label == "dynamic":
       #     sigma = self.get_sigma_episode(ep)
        if self.sigma_label == "dynamic_tmstp_epis":
            # cumulative timesteps across episodes
            sigma = self.get_sigma_t(t_ep)
        elif self.sigma_label == "dynamic_tmstp":
            # timestep for given episode
            sigma = self.get_sigma_t(t)
        elif self.sigma_label == "dynamic_cnt":
            # print("count_s", count_s)
            # cumulative counts for state s across episodes
            sigma = self.get_sigma_count(count_s)
        else:
            sigma = self.sigma_0
        return sigma

    def get_sigma_episode(self, episode):
        new_sigma = (0.95**episode) * self.sigma_0
        return new_sigma

    def get_sigma_t(self, t): # works for t being the time-step across episodes
        # or t being the time-step for a single episode
        new_sigma = (0.95**(t+1)) * self.sigma_0
        return new_sigma

    def get_sigma_count(self, count_s):
        sigma = (0.95 ** count_s) * self.sigma_0
        return sigma

    def run_Kris(self, env, run):
        np.random.seed(42 + run)
        action_space = env.action_space
        self.w = np.array([-0.001 * random.random() for _ in range(self.num_dims)])
        sigma = self.sigma_0 # initial value of sigma; sigma is passed from one episode to the other
        self.all_episodes_returns = []
        # ---- for debug ----------------
        count_exceeded_maxtime = 0
        cnt_terminal = 0
        # ---- end debug ----------------
        state_counter = CountState(self.sigma_0, env.num_states)
        t_ep = 0 # timestep across all episodes

        for episode in range(self.num_episodes):
            # print("-------------EPISODE:" + str(episode) + "--------------")
            self.rewards = []
            self.states = []
            self.actions = []
            self.mus = []
            self.sigmas_n = []
            self.rhos = []

            s = env.start()
            state_counter.increase_counts(s)
            a = self.chooseAction(s, action_space)
            self.states.append(s)
            self.actions.append(a)
            # Initialize:
            Rsum = 0.0
            T = float('inf')
            tao = 0
            t = 0 # timestep within each episode
            # sigma = self.get_sigma_episode(episode)
            self.sigmas_n += [sigma]
            self.rhos += [1.0] # behavior policy == target policy (by construction)
            done = False
            while tao < T-1:# for t in range(0, tao + 1): # +1 because of python
                if t < T:
                    r, s_p = env.step(s, a)  #s_p is an idx
                    self.states.append(s_p)
                    state_counter.increase_counts(s_p) # s must be an index
                    self.rewards.append(r)
                    Rsum += r
                    # ---- for debug ----------------
                    if t > self.num_timesteps:
                        count_exceeded_maxtime += 1
                        done = True
                    if s_p == env.terminal_state:
                        cnt_terminal += 1
                    # ---- end of debug -------------
                    if done or s_p == env.terminal_state:  # check if done according to environment transition, or done based on check above
                        T = t + 1
                        done = True
                    else:
                        a_p = self.chooseAction(s_p, action_space)
                        self.actions.append(a_p)
                        counts_s = int(state_counter.counts[s_p])
                        sigma = self.get_sigma(t, t_ep, counts_s)
                        self.sigmas_n.append(sigma) # should this contain n elements?
                        self.rhos.append(1.0)
                tao = t - self.n + 1
                if tao >= 0:
                    G = 0.0
                    m = min(t + 1, T)
                    count_k = 0
                    for k in range(min(t + 1, T), tao, -1):
                        if k == T:
                            G = self.rewards[T-1] # used to be G[T] -> changed it to T-1, because there is always one less reward than the number of states and actions (?)
                        else:
                            sigma_k = self.sigmas_n[k]
                            s_k = self.states[k]
                            a_k = self.actions[k]
                            Q_hat = self.q_estimate(s_k, a_k)
                            V = self.get_V(action_space, s_k)
                            target_policy = self.get_target_policy(s_k, action_space)
                            target_policy_sk_ak = target_policy[a_k]
                            G = self.rewards[k-1] + self.gamma * ( # used to be rewards[k]
                                sigma_k * self.rhos[k] +
                                (1 - sigma_k) * target_policy_sk_ak) * (
                                    G -
                                    Q_hat) + self.gamma * V
                    s_tao = self.states[tao]
                    a_tao = self.actions[tao]
                    self.update_weights(s_tao, a_tao, G)
                s = s_p
                a = a_p
                t += 1
                t_ep += 1
            self.all_episodes_returns.append(Rsum)
        return
