import random
import numpy as np
class NeuronNetwork():
    def __init__(self, network_sizes, numerOfEpochs, learning_rate=0.1, momentum = 0) -> None:
        self.number_of_layers = len(network_sizes)
        self.learning_rate = learning_rate
        # network_sizes -> liczba neuronow w poszczegolnych layerach
        self.network_sizes = network_sizes 
        self.momentum = momentum
        self.numerOfEpoch = numerOfEpochs
        self.weights = []
        self.number_of_neurons =0
        for i in range(0, network_sizes):
             self.number_of_neurons += network_sizes[i]
        for i in range(0, self.number_of_neurons):
            self.weights.append(random.uniform(-1,1)) 

     

def calculateError(self, target, output):
    error = 0
    for i in range(len(target)):
        error += (target[i] - output[i]) ** 2
    return error / 2
