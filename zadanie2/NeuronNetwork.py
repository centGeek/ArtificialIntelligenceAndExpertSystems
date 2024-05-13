import numpy as np
from functions import sigmoid_derivative, mean_squared_error
from layer import Layer
import matplotlib.pyplot as plt
from paths import PATH_TEST_LOGGING
from paths import PATH_STATS
import logging
logging.basicConfig(filename=PATH_STATS, level=logging.INFO)
class Network:
    def __init__(self, neurons_in_particular_layers: list, bias):
        self.num_layers = len(neurons_in_particular_layers)
        self.neurons_in_particular_layers = neurons_in_particular_layers
        self.input_num = neurons_in_particular_layers[0]
        self.layer_list = []
        self.bias = bias
        inputs_layer = self.input_num
        for i in range(1, self.num_layers-1):
            neuron_num_in_hidden_layer = neurons_in_particular_layers[i]
            self.layer_list.append(Layer(inputs_layer, neuron_num_in_hidden_layer, bias))
        inputs_layer = neuron_num_in_hidden_layer
        self.layer_list.append(Layer(inputs_layer, neurons_in_particular_layers[-1], bias))
        derivatives = [np.zeros((self.input_num, self.layer_list[0].neuron_num))]
        derivatives.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in enumerate(self.layer_list[:-1])])
        self.derivatives = derivatives

        activations = [np.zeros(self.input_num)] #liczba aktywacji taka sama
        activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.activations = activations

        weights = [np.zeros((self.input_num, self.layer_list[0].neuron_num))]
        weights.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                        enumerate(self.layer_list[:-1])])
        self.weights = weights
        self.old_weights = weights

        biases = []
        biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.biases = biases

        derivatives_biases = []
        derivatives_biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.derivatives_biases = derivatives_biases

    def forward(self, input_vector):
        self.activations[0] = np.array(input_vector)
        next_activation = self.activations[0]
        for x in range(len(self.layer_list)):
            next_activation = self.layer_list[x].forward(next_activation)
            self.activations[x + 1] = next_activation
            self.weights[x] = self.layer_list[x].get_weights()
            self.biases[x] = self.layer_list[x].get_biases()

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * sigmoid_derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)

            error = np.dot(delta, self.weights[x])

            self.derivatives_biases[x] = delta

    def update_weights(self, learning_rate, momentum):
        for i in range(len(self.weights)):
            derivatives = self.derivatives[i]
            if momentum != 0:
                current_weights = self.weights[i]
                self.weights[i] -= derivatives.T * learning_rate + (momentum * (current_weights - self.old_weights[i]))
                self.old_weights[i] = current_weights
            else:
                self.weights[i] -= derivatives.T * learning_rate
        self.update_neurons_weights()

    def update_biases(self, learning_rate):
        for i in range(len(self.biases)):
            derivatives_biases = self.derivatives_biases[i]
            self.biases[i] -= derivatives_biases * learning_rate
        self.update_neurons_biases()

    def update_neurons_weights(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_weights(self.weights[x])

    def update_neurons_biases(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_biases(self.biases[x])

    def train(self, input_data, epochs, learning_rate, jump, given_error, momentum):
        mistakes = []
        with open(PATH_TEST_LOGGING, "w") as file:
            for i in range(epochs):
                np.random.shuffle(input_data)
                sum_errors = 0
                for sample in input_data:
                    target = np.zeros_like(self.activations[-1])
                    g = int(sample[self.input_num])
                    target[g] = 1 # target jest tablica, przyjmuje 1 wartosc

                    # target = sample[self.input_num:] # target jest tablica, przyjmuje 1 wartosc 
                    self.forward(sample[:self.input_num])  

                    error = self.activations[-1] - target
                    self.back_propagation(error)

                    self.update_weights(learning_rate, momentum)

                    if self.bias:
                        self.update_biases(learning_rate)

                    sum_errors += mean_squared_error(target, self.activations[-1])
                if i % jump == 0:
                    file.write(f"{sum_errors / len(input_data)}\n")
                    mistakes.append(sum_errors / len(input_data))
                # if sum_errors / len(input_data) <= given_error:
                #     file.write(f"Uzyskano zadany poziom błędu w epoce {i}\n")
                #     file.write(f"błąd wynosi: {sum_errors / len(input_data)}\n")
                #     x_values = np.arange(0, i, 1)
                #     plt.plot(x_values, mistakes)
                #     plt.show()
                #     plt.title("Zależność błędu od ilości epok")
                #     plt.xlabel("Liczba epok")
                #     plt.ylabel("Wartość błędu")
                #     file.close()
                #     break
            x_values = np.arange(0, epochs, 1)
            plt.plot(x_values, mistakes)
            plt.title("Zależność błędu od ilości epok")
            plt.xlabel("Liczba epok")
            plt.ylabel("Wartość błędu")
            plt.show()


    def test(self, test_data):
        nr_of_outputs = self.neurons_in_particular_layers[-1]
        correctly_guessed = 0
        incorrectly_guessed = 0
        confusion_matrix = np.zeros((nr_of_outputs, nr_of_outputs))
        with open(PATH_TEST_LOGGING, "w") as file:
            for i, sample in enumerate(test_data):
                target = sample[self.input_num:]
                sample_input = sample[:self.input_num]
                self.forward(sample_input)
                output = self.activations[-1]

                predicted_class = np.argmax(output)                
                target_class = np.argmax(target)
                confusion_matrix[target_class, predicted_class] += 1


        confusion_string = ""
        for i in range(nr_of_outputs):
            confusion_string += f"class: {i}\n"
            tp = confusion_matrix[i, i]
            tn = 0
            fp = 0
            fn = 0
            for j in range(nr_of_outputs):
                for k in range(nr_of_outputs):
                    if k != i and j != i:
                        tn += confusion_matrix[j, k]

            for j in range(nr_of_outputs):
                if j != i:
                    fp += confusion_matrix[j, i]
                    fn += confusion_matrix[i, j]

            confusion_string += f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}\n"
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_measure = 2 * (precision * recall) / (precision + recall)
            confusion_string += f"Precision: {precision} Recall: {recall} F-measure: {f_measure} \n\n"
            logging.info(confusion_string)
            correctly_guessed += tp
            incorrectly_guessed += fp
        with open("training_effectiveness_info.txt", "w") as file:
            file.write(confusion_string)
            file.write(f"\n correctly classified: {correctly_guessed}\n")
            file.write(f"\n incorrectly classified: {incorrectly_guessed}\n")
