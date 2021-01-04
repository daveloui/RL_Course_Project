import csv
import os
import numpy as np
import matplotlib.pyplot as plt

aver_return_dic = {}

# fill these with appropriate values as specified in run_Experiments, or in generate_args
alphas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.7, 0.8, 0.9, 1.0] # alpha = 1/(tao * E[x.T * x])
sigmas = [0.0, 0.5, 1.0, "dynamic_tmstp_epis", "dynamic_cnt"] #dynamic is wrt episode index (as they do in the paper)
runs = 50

for filename in os.listdir():
	if ("Average_Rewards_all_Episodes_alpha_" in filename) and (".csv" in filename):
		parameters = filename.split("_run_")[0]
		parameters = parameters.split("Average_Rewards_all_Episodes_")[1]
		# print("parameters", parameters)
		if parameters not in aver_return_dic:
			aver_return_dic[parameters] = []

		with open(filename, 'r') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter = ',')
			mean_ret = float(list(csv_reader)[0][0]) # the first row contains the mean across episodes
			# print("mean return", mean_ret)
			assert isinstance(mean_ret, float)
			# print(aver_return_dic[parameters])
			aver_return_dic[parameters] += [mean_ret]

assert len(aver_return_dic[parameters]) == runs
# print(aver_return_dic)

for k, v_list in aver_return_dic.items():
	aver_return_dic[k] = np.mean(v_list)
	# print(type(aver_return_dic[k]))
# print(aver_return_dic)

# get data ready for plotting:
matrix = np.zeros((len(sigmas), len(alphas)))
for idx, (k, v) in enumerate(aver_return_dic.items()):
	# print("index", idx, "key", k, "val", v)
	for i, alpha in enumerate(alphas):
		if str(alpha) in k:
			for j, sigma in enumerate(sigmas):
				if str(sigma) in k:
					matrix[j, i] = v # rows are sigma_values, cols are alpha_values

# plot
fig, ax = plt.subplots(figsize=(14,8))
for i in range(len(matrix)): # numrows
	assert len(alphas) == len(matrix[i])
	plt.plot(alphas, matrix[i], label = ["sigma " + str(sigmas[i])])

ax.legend(loc='upper right')
ax.set_xlim(xmin=alphas[0], xmax=alphas[-1])
plt.xlabel('alphas')
plt.ylabel('Average Return over Episodes')
plt.title('Learning-Rate Sensitivity Curve')

plt.savefig("alpha_sens_exp2.png")
plt.close('all')
# for filename in os.listdir():
# 	if ("Average_Rewards_all_Episodes_alpha_" in filename) and (".csv" in filename):
# 		print("filename", filename)
# 		with open(filename, 'r') as csv_file:
# 			csv_reader = csv.reader(csv_file, delimiter = ',')
# 			mean_ret = float(list(csv_reader)[0][0]) # the first row contains the mean across episodes
# 			# print("mean_ret", mean_ret)
# 			for alpha in alphas:
# 				if "alpha_" + str(alpha) in filename:
# 					print("alpha", alpha)
# 					for sigma in sigmas:
# 						if "sigma_" + str(sigma) in filename:
# 							print("mean_ret", mean_ret)
# 							dic[sigma] += [mean_ret]
# 			# print(dic)
# print(dic)
