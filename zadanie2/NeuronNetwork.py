import random
import numpy as np
import logging
logging.basicConfig(filename='training_logs.log', level=logging.INFO)
class NeuronNetwork():
    def __init__(self, neuron_in_particular_layers: list, useBias: bool) -> None:
        self.num_layers = len(neuron_in_particular_layers)
        self.neuron_in_particular_layers = neuron_in_particular_layers
        self.useBias = useBias
        self.biases = []
        self.weights = []
        self.velocity = []
        if useBias:
            for layer_index in range(1, len(neuron_in_particular_layers)):
                layer_bias = np.random.uniform(-1, 1, (neuron_in_particular_layers[layer_index], 1))
                self.biases.append(layer_bias)
        else:
            for i in range(1, len(neuron_in_particular_layers)):
                bias_matrix = np.zeros((neuron_in_particular_layers[i], 1))
                self.biases.append(bias_matrix)
        for x, y in zip(neuron_in_particular_layers[:-1], neuron_in_particular_layers[1:]):
            weight_matrix = np.random.uniform(-1, 1, (y, x))
            self.weights.append(weight_matrix)
        for w in self.weights:
            velocity_matrix = np.zeros(w.shape)
            self.velocity.append(velocity_matrix)
        # zobacz jak wygladaja poszczegolne poczatkowe parametry sieci
        logging.info(f"Initialized velocity {self.velocity}")
        logging.info(f"Initialized biases {self.biases}")
        logging.info(f"Initialized weights {self.weights}")     
    def update_weights(self, new_weights):
        for x in range(len(self.weights)):
            self.weights[x] = new_weights[x]

