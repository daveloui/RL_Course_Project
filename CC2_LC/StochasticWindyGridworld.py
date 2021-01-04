import random
from WindyGridworld import WindyGridworld


class StochasticWindyGridworld(WindyGridworld):
    def __init__(self, x_dim=10, y_dim=7, wind_strengths=None, random_move_probability=0.1):
        super().__init__(x_dim, y_dim, wind_strengths)
        self.random_move_probability = random_move_probability

    def step(self, S, A):
        if random.random() < self.random_move_probability:
            return self.__make_random_move(S)
        else:
            return super().step(S, A)

    def __make_random_move(self, S):
        random_value = random.random()
        # Go left
        if random_value < 0.125:
            return self.left(S)
        # Go up-left
        elif 0.25 > random_value >= 0.125:
            _, Snext = self.left(S)
            return self.up(Snext)
        # Go up
        elif 0.375 > random_value >= 0.25:
            return self.up(S)
        # Go up-right
        elif 0.5 > random_value >= 0.375:
            _, Snext = self.up(S)
            return self.right(Snext)
        # Go right
        elif 0.625 > random_value >= 0.5:
            return self.right(S)
        # Go down-right
        elif 0.75 > random_value >= 0.625:
            _, Snext = self.down(S)
            return self.right(Snext)
        # Go down
        elif 0.875 > random_value >= 0.75:
            return self.down(S)
        # Go down-left
        else:
            _, Snext = self.left(S)
            return self.down(Snext)
