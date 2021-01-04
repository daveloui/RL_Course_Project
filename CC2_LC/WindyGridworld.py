# Courtesy of Zachary Holland; some modifications were made to his code
# Need to specify when done=True

import numpy as np
import scipy
import math


class WindyGridworld():
    """
    wind_strengths is a list of integers >=0 which represents the strengths of the wind at each x position.
    """
    def __init__(self, x_dim=10, y_dim=7, wind_strengths=None):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_shape = (x_dim, y_dim)
        self.num_states = x_dim * y_dim
        self.num_actions = 4
        self.action_space = np.arange(self.num_actions)

        if wind_strengths is None:
            self.wind_strengths = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind_strengths = wind_strengths
        self.terminal_state = self.coords_to_index(self.x_dim-3, int(self.y_dim / 2) + 1)

    def start(self):
        return self.coords_to_index(0, int(self.y_dim / 2) + 1)

    def step(self, S, A): # s and a are indices of the state, action
        if A == 0:
            return self.left(S)
        elif A == 1:
            return self.up(S)
        elif A == 2:
            return self.right(S)
        elif A == 3:
            return self.down(S)

    def index_to_coords(self, S):
        return int(S % self.x_dim), int(S // self.x_dim)

    def coords_to_index(self, x, y):
        return int(y*self.x_dim + x)

    def left(self, S):
        x, y = self.index_to_coords(S)
        new_x = max([x-1, 0])
        new_y = min([y+self.wind_strengths[new_x], self.y_dim-1])
        return -1, self.coords_to_index(new_x, new_y)

    def up(self, S):
        x, y = self.index_to_coords(S)
        return -1, self.coords_to_index(x, min([y+1+self.wind_strengths[x], self.y_dim-1]))

    def right(self, S):
        x, y = self.index_to_coords(S)
        new_x = min([x+1, self.x_dim-1])
        new_y = min([y+self.wind_strengths[new_x], self.y_dim-1])
        return -1, self.coords_to_index(new_x, new_y)

    def down(self, S):
        x, y = self.index_to_coords(S)
        new_y = max([min([y-1+self.wind_strengths[x], self.y_dim-1]), 0])
        return -1, self.coords_to_index(x, new_y)
