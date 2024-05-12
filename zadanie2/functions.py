import numpy as np
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def sigmoid_derivative(x):
    return x * (1.0 - x)
def mean_squared_error(target, output):
    return np.average((output - target) ** 2)
def calculateError(self, target, output):
    error = 0
    for i in range(len(target)):
        error += (target[i] - output[i]) ** 2
    return error / 2

