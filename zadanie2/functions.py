import numpy as np
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def sigmoid_derivative(x):
    return x * (1.0 - x)
def mean_squared_error(target, output):
    return np.average((output - target) ** 2)

