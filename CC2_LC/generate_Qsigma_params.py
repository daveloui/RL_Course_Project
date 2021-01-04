alphas = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 0.7, 0.8, 0.9, 1.0] # alpha = 1/(tao * E[x.T * x])
sigmas = [0.0, 0.5, 1.0, 3.0, 5.0] #dynamic is wrt episode index (as they do in the paper)
num_runs = 50
permutation_file = open("params_Qsigma_exp2.txt", "a")


for alpha in alphas:
	for sigma in sigmas:
		for i in range(num_runs):
			job_str = "python3 run_experiment.py " + str(alpha) + " " + str(sigma) + " " + str(i) + "\n"
			permutation_file.write(job_str)
