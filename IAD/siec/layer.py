
import numpy as np
from neuron import Neuron
from functions import sigmoid, sigmoid_derivative


class Layer:
    def __init__(self, input_num: int, neuron_num: int, bias: bool,
                 activation=sigmoid, derivative=sigmoid_derivative, rng=None):
        self.neuron_num = neuron_num
        self.input_num = input_num
        self.bias = bias
        self.activation = activation
        self.derivative = derivative
        self.neurons = np.array([Neuron(input_num, bias, rng) for _ in range(neuron_num)])
        self.output = None

    def forward(self, inputs):
        self.output = np.array([neuron.forward(inputs, self.activation)
                                for neuron in self.neurons])
        return self.output

    def weighted_sum(self):
        return np.array([neuron.weighted_sum for neuron in self.neurons])

    def update_weights(self, new_weights):
        for x in range(len(self.neurons)):
            self.neurons[x].update_weight(new_weights[x])

    def get_weights(self):
        return np.array([neuron.weights for neuron in self.neurons])

    def get_biases(self):
        return np.array([neuron.bias for neuron in self.neurons])

    def update_biases(self, new_biases):
        for x in range(len(self.neurons)):
            self.neurons[x].update_bias(new_biases[x])

