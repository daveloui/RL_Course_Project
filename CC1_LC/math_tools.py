import numpy as np

def argMax(arr):
    indices = np.where(arr == np.max(arr))[0]
    if len(indices) < 1:
        print(arr)
        raise ArithmeticError()

    return np.random.choice(indices)
