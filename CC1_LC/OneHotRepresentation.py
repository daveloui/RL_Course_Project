import numpy as np
import scipy

class OneHot():
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_dims = self.num_states * self.num_actions
        self.x = np.zeros(self.num_dims)

    def convert(self, s_idx, a_idx): # s and a are indices
        one_idx = s_idx + self.num_states * a_idx
        x = np.copy(self.x)
        x[one_idx] = 1.0
        self.check_one_hot(x)
        return x, one_idx

    def check_one_hot(self, vec):
        cnt = 0
        for v_i in vec:
            if v_i == 1.0:
                cnt += 1
        assert cnt == 1.0
        return
