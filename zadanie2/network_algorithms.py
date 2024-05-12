import random
import numpy as np
import pickle
from NeuronNetwork import NeuronNetwork
import matplotlib as plt
from functions import mean_squared_error
from functions import sigmoid, sigmoid_derivative


class Network(object):
    def __init__(self, network: NeuronNetwork):
        self.network = network
    
    def forward(self, a):
        for i in range(len(self.biases)):
            bias = self.biases[i]
            weight = self.weights[i]
            a = sigmoid(np.dot(weight, a) + bias)
        return a

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * sigmoid_derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)
            activations = [np.zeros(input_num)]
            activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
            self.activations = activations
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
    def train(self, input_data, epochs, learning_rate, jump, given_error, shuffle, momentum):
        mistakes = []
        with open("training_information.txt", "w") as file:
            for i in range(epochs):
                if shuffle:
                    np.random.shuffle(input_data)
                sum_errors = 0
                for sample in input_data:
                    target = sample[self.input_num:]     
                    self.forward(sample[:self.input_num])

                    error = self.activations[-1] - target
                    self.back_propagation(error)

                    self.update_weights(learning_rate, momentum)

                    if self.network.biases:
                        self.update_biases(learning_rate)

                    sum_errors += mean_squared_error(target, self.activations[-1])
                if i % jump == 0:
                    file.write(f"{sum_errors / len(input_data)}\n")
                    mistakes.append(sum_errors / len(input_data))
                if sum_errors / len(input_data) <= given_error:
                    file.write(f"Uzyskano zadany poziom błędu w epoce {i}\n")
                    file.write(f"błąd wynosi: {sum_errors / len(input_data)}\n")
                    x_values = np.arange(0, i, 1)
                    plt.plot(x_values, mistakes)
                    plt.show()

                    file.close()
                    break
            x_values = np.arange(0, epochs, 1)
            plt.plot(x_values, mistakes)
            plt.show()


    def test(self, test_data):
        nr_of_outputs = test_data.shape[1] - self.input_num
        correctly_guessed = 0
        incorrectly_guessed = 0
        confusion_matrix = np.zeros((nr_of_outputs, nr_of_outputs))
        with open("testing_information.txt", "w") as file:
            for i, sample in enumerate(test_data):
                target = sample[self.input_num:]
                sample_input = sample[:self.input_num]
                self.forward(sample_input)
                output = self.activations[-1]

                predicted_class = np.argmax(output)
                target_class = np.argmax(target)
                confusion_matrix[target_class, predicted_class] += 1

                file.write(f"wzorzec numer: {i}, {sample_input}\n")
                file.write(f"popełniony błąd: {mean_squared_error(target, output)}\n")
                file.write(f"pożądany wzorzec odpowiedzi: {target}\n")
                for x in range(len(output)):
                    file.write(f"błąd popełniony na {x} wyjściu: {output[x] - target[x]}\n")
                for x in range(len(output)):
                    file.write(f"wartość na {x} wyjściu: {output[x]}\n")
                file.write(f"wartości wag neuronów wyjściowych\n {self.weights[-1]}\n")
                for x in reversed(range(1, len(self.activations) - 1)):
                    file.write(f"wartości wyjściowych neuronów ukrytych, warstwa {x}:\n {self.activations[x]}\n")
                for x in reversed(range(len(self.weights) - 1)):
                    file.write(f"wartości wag neuronów ukrytych, warstwa {x + 1}:\n {self.weights[x]}\n\n\n")
        file.close()

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
            correctly_guessed += tp
            incorrectly_guessed += fp
        with open("training_effectiveness_info.txt", "w") as file:
            file.write(confusion_string)
            file.write(f"\n correctly classified: {correctly_guessed}\n")
            file.write(f"\n incorrectly classified: {incorrectly_guessed}\n")
