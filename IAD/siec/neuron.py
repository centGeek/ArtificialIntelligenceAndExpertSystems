import numpy as np
from functions import sigmoid


class Neuron:
    def __init__(self, inputs_num: int, bias: bool):
        self.neuron_inputs = None
        self.weights = np.random.uniform(low=-1.0, high=1.0, size=inputs_num)
        if bias:
            self.bias = np.random.uniform(low=-1.0, high=1.0)
        else:
            self.bias = 0
        self.output = 0
        self.weighted_sum = 0

    def forward(self, inputs, activation_func=sigmoid) -> float:
        self.weighted_sum = np.dot(self.weights, inputs) + self.bias
        self.output = activation_func(self.weighted_sum)
        return self.output

    def loss_function(self, true_value):
        return np.mean((self.output - true_value) ** 2)

    def update_weight(self, new_weights):
        self.weights = new_weights

    def update_bias(self, new_bias):
        self.bias = new_bias
