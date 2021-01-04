import numpy as np
import sys
import random
import csv
from multiprocessing import Pool

class Variance_World:
	def __init__(self, iteration, run_num, total_time, alpha_ext, gamma_ext, alpha_int, gamma_int, num_funcs, theta, target_noise, multiplier, root):
		rand_seed = random.randint(1,10000)
		np.random.seed(rand_seed)

		# Experiment
		self.iteration = iteration
		self.run_num = run_num
		self.total_time = total_time

		# Environment
		self.num_states = 11
		self.num_actions = 2

		# RND
		self.num_funcs = num_funcs
		self.theta = theta
		self.target_noise = target_noise
		self.multiplier = multiplier
		self.root = root

		self.target = np.random.randn(self.num_states, self.num_actions, self.num_funcs)
		self.predictor = np.random.randn(self.num_states, self.num_actions, self.num_funcs)

		# SARSA
		self.alpha_ext = alpha_ext
		self.gamma_ext = gamma_ext
		self.alpha_int = alpha_int
		self.gamma_int = gamma_int

		self.q_ext = [[0.0 for i in range(self.num_actions)] for j in range(self.num_states)]

		if root:
			initial_q_int = 2.0**(0.5) / (1.0 - self.gamma_int)
		else:
			initial_q_int = 2.0 / (1.0 - self.gamma_int)

		self.q_int = [[initial_q_int for i in range(self.num_actions)] for j in range(self.num_states)]

		# Statistics
		self.statistics = []
		self.state_action_visits = [[0 for i in range(self.num_actions)] for j in range(self.num_states)]
		self.big_reward_visits = 0

	def run(self):
		overall_time = 1
		episode_time = 1
		state, action = self.initialise_episode()
		
		while overall_time < self.total_time:
			next_state, reward, terminal = self.take_action(state, action, episode_time)

			bonus = self.get_bonus(state, action)

			next_action = self.choose_action(next_state)

			# Should be overall_time - not episode_time
			self.update_predictor(state, action, overall_time)

			self.q_ext[state][action] = self.q_ext[state][action] + self.alpha_ext * (reward + (self.gamma_ext * self.q_ext[next_state][next_action]) - self.q_ext[state][action])

			# If at a terminal state, we need the initial state and action from the next episode to update Q_int
			if terminal:
				next_state, next_action = self.initialise_episode()
				self.q_int[state][action] = self.q_int[state][action] + self.alpha_int * (bonus + (self.gamma_int * self.q_int[next_state][next_action]) - self.q_int[state][action])
				episode_time = 1
			else:
				self.q_int[state][action] = self.q_int[state][action] + self.alpha_int * (bonus + (self.gamma_int * self.q_int[next_state][next_action]) - self.q_int[state][action])
				episode_time += 1

			state = next_state
			action = next_action

			if overall_time == 100 or overall_time == 1000 or overall_time == 10000:
				self.process_statistics(overall_time)

			overall_time += 1

		self.process_statistics(overall_time)

	def process_statistics(self, timesteps):
		with open("/local/data/trudeau1/Variance_World/Iteration_" + str(self.iteration) + "_short/alpha_ext_" + str(self.alpha_ext) + "_gamma_ext_" + str(self.gamma_ext) + "_alpha_int_" + str(self.alpha_int) + "_gamma_int_" + str(self.gamma_int) + "_theta_" + str(self.theta) + "_target_noise_" + str(self.target_noise) + "_multiplier_" + str(self.multiplier) + "_root_" + str(self.root) + "_run_" + str(self.run_num) + "_SGD_Tabular_ShortVarianceWorld_Statistics_" + str(timesteps) + ".csv", mode='a') as rnd_file:
			rnd_writer = csv.writer(rnd_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for row in self.statistics:
				rnd_writer.writerow(row)

		with open("/local/data/trudeau1/Variance_World/Iteration_" + str(self.iteration) + "_short/alpha_ext_" + str(self.alpha_ext) + "_gamma_ext_" + str(self.gamma_ext) + "_alpha_int_" + str(self.alpha_int) + "_gamma_int_" + str(self.gamma_int) + "_theta_" + str(self.theta) + "_target_noise_" + str(self.target_noise) + "_multiplier_" + str(self.multiplier) + "_root_" + str(self.root) + "_run_" + str(self.run_num) + "_SGD_Tabular_ShortVarianceWorld_Q_int_" + str(timesteps) + ".csv", mode='a') as rnd_file:
			rnd_writer = csv.writer(rnd_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for state in range(self.num_states):
				rnd_writer.writerow(self.q_int[state])

		with open("/local/data/trudeau1/Variance_World/Iteration_" + str(self.iteration) + "_short/alpha_ext_" + str(self.alpha_ext) + "_gamma_ext_" + str(self.gamma_ext) + "_alpha_int_" + str(self.alpha_int) + "_gamma_int_" + str(self.gamma_int) + "_theta_" + str(self.theta) + "_target_noise_" + str(self.target_noise) + "_multiplier_" + str(self.multiplier) + "_root_" + str(self.root) + "_run_" + str(self.run_num) + "_SGD_Tabular_ShortVarianceWorld_Q_ext_" + str(timesteps) + ".csv", mode='a') as rnd_file:
			rnd_writer = csv.writer(rnd_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for state in range(self.num_states):
				rnd_writer.writerow(self.q_ext[state])

		with open("/local/data/trudeau1/Variance_World/Iteration_" + str(self.iteration) + "_short/alpha_ext_" + str(self.alpha_ext) + "_gamma_ext_" + str(self.gamma_ext) + "_alpha_int_" + str(self.alpha_int) + "_gamma_int_" + str(self.gamma_int) + "_theta_" + str(self.theta) + "_target_noise_" + str(self.target_noise) + "_multiplier_" + str(self.multiplier) + "_root_" + str(self.root) + "_run_" + str(self.run_num) + "_SGD_Tabular_ShortVarianceWorld_Visits_" + str(timesteps) + ".csv", mode='a') as rnd_file:
			rnd_writer = csv.writer(rnd_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for state in range(self.num_states):
				rnd_writer.writerow(self.state_action_visits[state])


	def initialise_episode(self):
		state = np.random.choice(range(4, 7, 1))
		action = self.choose_action(state)

		return state, action

	def take_action(self, state, action, episode_time):
		if action == 0:
			next_state = state - 1
		elif action == 1:
			next_state = state + 1

		# Left, constant small reward
		if next_state == 0:
			reward = 0.01 / np.power(self.gamma_ext, 5)
			terminal = True
		elif next_state == 10:
			rewards = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 10.0]
			reward = np.random.choice(rewards) / np.power(self.gamma_ext, 5)
			terminal = True
			self.big_reward_visits += 1
		else:
			reward = 0.0
			terminal = False

		if terminal:
			self.statistics += [[episode_time, self.big_reward_visits]]

		return next_state, reward, terminal


	def choose_action(self, state):
		best_action = random.choice(range(self.num_actions))
		best_Q = self.q_ext[state][best_action] + self.multiplier*self.q_int[state][best_action]

		for i in range(self.num_actions):
			if self.q_ext[state][i] + self.multiplier*self.q_int[state][i] > best_Q:
				best_Q = self.q_ext[state][i] + self.multiplier*self.q_int[state][i]
				best_action = i

		self.state_action_visits[state][best_action] += 1

		return best_action

	def get_bonus(self, state, action):
		error = self.predictor[state][action] - self.target[state][action]
		bonus = np.mean(error**(2.0))

		if self.root:
			bonus = bonus**(0.5)

		return bonus

	def update_predictor(self, state, action, overall_time):
		lr = self.theta / float(overall_time)
		# Trying constant step-size
		# lr = self.theta

		noisy_target = np.random.multivariate_normal(self.target[state][action], self.target_noise * np.identity(self.num_funcs))
		error = self.predictor[state][action] - noisy_target

		gradient = (2.0 / float(self.num_funcs)) * error

		self.predictor[state][action] = self.predictor[state][action] - lr*gradient


def run_experiments(run_args):
	total_time = 100000
	num_funcs = 30
	gamma_ext = 0.99
	runs = 30

	iteration = run_args[0]
	alpha_ext = run_args[1]
	alpha_int = run_args[2]
	gamma_int = run_args[3]
	theta = run_args[4]
	target_noise = run_args[5]
	multiplier = run_args[6]
	root = run_args[7]

	for i in range(runs):
		variance_world = Variance_World(iteration, i, total_time, alpha_ext, gamma_ext, alpha_int, gamma_int, num_funcs, theta, target_noise, multiplier, root)
		variance_world.run()


if __name__ == "__main__":
	# (self, episodes, alpha, gamma, num_funcs, theta, target_noise, multiplier, root):

	# Root False, No Noise
	alpha_exts = [0.0001, 0.001, 0.01, 0.03]
	alpha_ints = [0.001, 0.01, 0.03, 0.1]
	gamma_ints = [0.05, 0.1, 0.2, 0.4, 0.8]
	thetas = [30.0, 300.0]
	target_noises = [0.0]
	multipliers = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
	roots = [False]

	# Root True, No Noise
	alpha_exts = [0.0001, 0.001, 0.01]
	alpha_ints = [0.001, 0.01, 0.03, 0.1]
	gamma_ints = [0.05, 0.1, 0.2, 0.4, 0.8]
	thetas = [30.0, 300.0]
	target_noises = [0.0]
	multipliers = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
	roots = [True]

	# Root False, Noise
	alpha_exts = [0.0001, 0.001, 0.01]
	alpha_ints = [0.001, 0.01, 0.03, 0.1]
	gamma_ints = [0.05, 0.1, 0.2, 0.4, 0.8]
	thetas = [30.0, 300.0]
	target_noises = [1.0]
	multipliers = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
	roots = [False]

	# Root True, Noise
	alpha_exts = [0.0001, 0.001, 0.01]
	alpha_ints = [0.001, 0.01, 0.03, 0.1]
	gamma_ints = [0.1, 0.2, 0.4, 0.8, 0.99, 0.999]
	thetas = [30.0, 300.0]
	target_noises = [1.0]
	multipliers = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
	roots = [True]

	iterations = 1

	permutations = []

	for iteration in range(iterations):
		for alpha_ext in alpha_exts:
			for alpha_int in alpha_ints:
				for gamma_int in gamma_ints:
					for theta in thetas:
						for target_noise in target_noises:
							for multiplier in multipliers:
								for root in roots:
									permutations += [(iteration, alpha_ext, alpha_int, gamma_int, theta, target_noise, multiplier, root)]

	threads = 12
	pool = Pool(threads)

	for j in range(0, len(permutations), threads):
		pool.map(run_experiments, permutations[j:j+threads])





