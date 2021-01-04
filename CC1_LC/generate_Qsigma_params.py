alphas = [0.4] # alpha = 1/(tao * E[x.T * x])
sigmas = [0.0, 0.5, 1.0, 2.0] #dynamic is wrt episode index (as they do in the paper)
num_runs = 50
permutation_file = open("params_Qsigma_LC.txt", "a")


for alpha in alphas:
	for sigma in sigmas:
		for i in range(num_runs):
			job_str = "python3 run_experiment.py " + str(alpha) + " " + str(sigma) + " " + str(i) + "\n"
			permutation_file.write(job_str)
