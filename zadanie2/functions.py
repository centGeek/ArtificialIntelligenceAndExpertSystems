import numpy as np
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
def derivative_sigmoid(z):
    s = sigmoid(z)
    return s * (1-s)
