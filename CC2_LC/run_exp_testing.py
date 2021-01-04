from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import random
import csv
import cProfile
import sys
from multiprocessing import Pool, Lock

# from N_Step_Sarsa import N_Step_Sarsa
from Q_sigma import Q_sigma
from WindyGridworld import WindyGridworld
from StochasticWindyGridworld import StochasticWindyGridworld
from OneHotRepresentation import OneHot


def init(l):
    global lock
    lock = l

# later: find a way to have if else for dealing with both types of environments
def run_experiments(run_args):
    alpha = run_args[0]
    sigma = run_args[1]
    run_num = run_args[2]
    n = 3
    gamma = 1.0
    epsilon = 0.1
    num_episodes = 100 #100 as the paper did
    num_timesteps = 10000
    sigmas_dict = {"dynamic" : 2.0, "dynamic_tmstp_epis" : 3.0, "dynamic_tmstp" : 4.0, "dynamic_cnt" : 5.0}
    # dynamic (sigma changes according to episode index)
    # --> dynamic_tmstp_epis: sigma changes wrt cumulative timesteps across episodes
    # dynamic_tmstp: sigma changes wrt timestep for given episode
    # --> dynamic_cnt: sigma changes wrt cumulative counts for state s across episodes
    for item in sigmas_dict.items():
        k, v = item[0], item[1]
        if sigma == v:
            sigma = k
    # print("sigma after transformation", sigma)

    env = StochasticWindyGridworld()
    oneHot = OneHot(env.num_states, env.num_actions)

    print("run_num", run_num)
    solution_params = {"n":n, "alpha":alpha, "epsilon":epsilon, "sigma":sigma}
    problem_params = {"gamma":gamma, "num_episodes":num_episodes, "num_timesteps":num_timesteps}

    q_sigma = Q_sigma(solution_params, problem_params, oneHot, type_rep="one_hot")
    q_sigma.run_Kris(env, run_num)
    e_return = q_sigma.all_episodes_returns
    mean_e_return = np.mean(e_return)
    print("undisc return for run =", mean_e_return)


    with open("alpha_sensitivity_results_exp2/Average_Rewards_all_Episodes_" + "alpha_" +
          str(alpha) + "_sigma_" + str(sigma) + "_run_" + str(run_num) + ".csv",
          mode='a') as Qsigma_file:
          Qsigma_writer = csv.writer(Qsigma_file, delimiter=",", quotechar='"',
            quoting=csv.QUOTE_MINIMAL)
          Qsigma_writer.writerow([mean_e_return])
          Qsigma_writer.writerow(e_return)



if __name__ == "__main__":

    # alphas = [0.01, 0.02]
    # sigmas = [1.0, 3.0] #dynamic is wrt episode index (as they do in the paper)
    alphas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.7, 0.8, 0.9, 1.0] # alpha = 1/(tao * E[x.T * x])
    sigmas = [0.0, 0.5, 1.0, 3.0, 5.0] #dynamic is wrt episode index (as they do in the paper)
    num_runs = 50 # 50

    # Multiprocessing
    number_of_cores = 4
    lock = Lock()
    pool = Pool(number_of_cores, initializer=init, initargs=(lock, ))
    permutations = []

    for alpha in alphas:
        for sigma in sigmas:
            for i in range(num_runs):
                permutations += [(alpha, sigma, i)]
    # print("permutations", permutations)

    for j in range(0, len(permutations), number_of_cores):
        pool.map(run_experiments, permutations[j:j + number_of_cores])
    pool.close()
    pool.join()
