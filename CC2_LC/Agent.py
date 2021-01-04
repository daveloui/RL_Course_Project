# Fatima Davelouis
# Todo: move each agent to its own python file
# see what is common between them and add it to the generic agent class
# see what is common between each agent's run methods and add them to a generic run method?
# or to a run method in run_experiment.py?

import numpy as np
import scipy
import random
from math_tools import argMax


class LearningAgent:
    def __init__(self, solution_params, problem_params, s_a_representation, type_rep="one_hot"):
        if type_rep == "tc":
            self.totalTiles = solution_params["totalTiles"] # = num_dims? Or number of tiles per tiling?
            self.num_tilings = solution_params["num_tilings"]
            self.tileCoder = s_a_representation
            self.num_dims = self.totalTiles + 1 # check this
        elif type_rep == "one_hot":
            self.oneHot = s_a_representation # the object
            self.num_dims = self.oneHot.num_dims

        self.n = solution_params["n"]
        self.alpha = solution_params["alpha"]
        self.epsilon = solution_params["epsilon"]
        self.sigma = solution_params["sigma"]

        self.gamma = problem_params["gamma"]
        self.num_episodes = problem_params["num_episodes"]
        self.num_timesteps = problem_params["num_timesteps"]


    def stateToFeatures(self, state, action, done):
        # state is a state-idx and action is an action-idx
        if done:
            currX = np.zeros(self.num_dims)
            idx = None
        else:
            currX, idx = self.oneHot.convert(state, action)
        return currX, idx

    def start(self, s):
        raise NotImplementedError

    def update_weights(self, s, a, G):
        # Linear representation: the gradient of Q(s, a) is x(s, a)
        self.q_hat = self.q_estimate(s, a)
        self.w = self.w + self.alpha * (G - self.q_hat) * self.x

    def q_estimate(self, s, a):
        # Linear representation
        self.x, idx = self.stateToFeatures(s, a, False)
        q_estimate = self.w.T.dot(self.x)
        return q_estimate

    def getQ(self, state, action_space):
        # returns Q-estimate for all actions, with fixed state
        Q = []
        for a in action_space:
            currQ = self.q_estimate(state, a)
            Q.append(currQ)
        assert len(Q) == len(action_space)
        return Q

# there might be issues here, where the target policy != beh policy?
# make this end with self.policy?
    def get_target_policy(self, state, action_space):
        Q = self.getQ(state, action_space)
        greedy_action = argMax(Q)
        policy = np.repeat(self.epsilon / len(Q), len(Q))
        policy[greedy_action] += 1 - self.epsilon
        return policy

    def get_V(self, action_space, s_k):
        V = 0.0
        target_policy = self.get_target_policy(s_k, action_space)
        for a in action_space: # shouldn't need to use enumerate
            V += target_policy[a] * self.q_estimate(s_k, a)
        return V

    def get_beh_policy(self, state, action_space):
        self.policy = self.get_target_policy(state, action_space)
        return self.policy

    # make this method use self.policy?
    def chooseAction(self, state, action_space):
    	# Choosing random action in the case where rand < epsilon
        beh_policy = self.get_beh_policy(state, action_space)
        action_chosen = np.random.choice(
          action_space,
          p=beh_policy
        )
        return action_chosen
