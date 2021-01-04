from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import random
import csv
import cProfile
import sys
# from multiprocessing import Pool, Lock


from Q_sigma import Q_sigma
from WindyGridworld import WindyGridworld
from StochasticWindyGridworld import StochasticWindyGridworld
from OneHotRepresentation import OneHot



# later: find a way to have if else for dealing with both types of environments
def run_experiments(run_args):
    alpha = run_args[0]
    sigma = run_args[1]
    run_num = run_args[2]
    # constant parameters (taken from the paper) :
    n = 4
    gamma = 1.0
    epsilon = 0.1
    num_episodes = 100
    num_timesteps = 10000

    sigmas_dict = {"dynamic" : 2.0, "dynamic_tmstp_epis" : 3.0, "dynamic_tmstp" : 4.0, "dynamic_cnt" : 5.0}
    for item in sigmas_dict.items():
        k, v = item[0], item[1]
        if sigma == v:
            sigma = k
    # env = WindyGridworld()
    env = StochasticWindyGridworld()
    oneHot = OneHot(env.num_states, env.num_actions)

    print("run_num", run_num)
    solution_params = {"n":n, "alpha":alpha, "epsilon":epsilon, "sigma":sigma}
    problem_params = {"gamma":gamma, "num_episodes":num_episodes, "num_timesteps":num_timesteps}

    q_sigma = Q_sigma(solution_params, problem_params, oneHot, type_rep="one_hot")
    q_sigma.run_Kris(env, run_num)
    all_epis_rewards = q_sigma.all_rewards # a list of E lists. Each list has some number of cumlative rewards for that episode.

    filename = str("LC_results/Average_Rewards_all_Episodes_" + "alpha_" +
          str(alpha) + "_sigma_" + str(sigma) + "_run_" + str(run_num) + ".npy")
    np.save(filename, all_epis_rewards)


if __name__ == "__main__":
    alpha = float(sys.argv[1])
    sigma = float(sys.argv[2])
    run_num = int(sys.argv[3])
    run_experiments((alpha, sigma, run_num))
