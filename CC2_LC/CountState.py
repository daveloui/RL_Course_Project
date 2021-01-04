import numpy as np

class CountState():
    def __init__(self, sigma_0, num_states):
        self.sigma_0 = sigma_0
        self.num_states = num_states
        self.counts = np.zeros(num_states)

    def increase_counts(self, state_idx): # state is an index
        self.counts[state_idx] += 1
