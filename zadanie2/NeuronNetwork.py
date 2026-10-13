import copy

import numpy as np
from functions import sigmoid_derivative, mean_squared_error
from layer import Layer
import matplotlib.pyplot as plt
from paths import PATH_TEST_LOGGING, PATH_STATS, PATH_TRAIN_EFFECTS
import pandas as pd
import logging
import os


def _create_logger(name, path):
    """Osobny logger z wlasnym plikiem - basicConfig dziala tylko raz na proces."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


stats_logger = _create_logger("statystyki", PATH_STATS)
effects_logger = _create_logger("skutecznosc", PATH_TRAIN_EFFECTS)


class Network:
    def __init__(self, neurons_in_particular_layers: list, bias):
        self.num_layers = len(neurons_in_particular_layers)
        self.neurons_in_particular_layers = neurons_in_particular_layers
        self.input_num = neurons_in_particular_layers[0]
        self.layer_list = []
        self.bias = bias
        inputs_layer = self.input_num

        for i in range(1, self.num_layers - 1):
            neuron_num_in_hidden_layer = neurons_in_particular_layers[i]
            self.layer_list.append(Layer(inputs_layer, neuron_num_in_hidden_layer, bias))
            inputs_layer = neuron_num_in_hidden_layer
        self.layer_list.append(Layer(inputs_layer, neurons_in_particular_layers[-1], bias))

        derivatives = [np.zeros((self.input_num, self.layer_list[0].neuron_num))]
        derivatives.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                            enumerate(self.layer_list[:-1])])
        self.derivatives = derivatives
        activations = [np.zeros(self.input_num)]  # liczba aktywacji taka sama
        activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.activations = activations

        # Bufory wag startuja z rzeczywistymi wagami neuronow. Wczesniej byly zerami,
        # wiec w pierwszym kroku momentum diff = old_weights - weights = -W0 i zamiast
        # W1 = W0 - lr*grad wychodzilo W1 = (1 + momentum) * W0 - lr*grad, czyli skok
        # przeskalowujacy wszystkie wagi sieci (dla momentum=0.9 az 1.9-krotnie).
        self.weights = [layer.get_weights() for layer in self.layer_list]
        self.old_weights = [w.copy() for w in self.weights]

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

                remember = copy.deepcopy(self.weights[i])
                diff = self.old_weights[i] - self.weights[i]
                self.weights[i] -= derivatives.T * learning_rate + (momentum * diff)

                self.old_weights[i] = remember

                #print("sdasd")
                # current_weights = self.weights[i]
                #
                # dif = self.weights[i] - self.old_weights[i]
                # subtractCalc = (derivatives.T * learning_rate) + (momentum * dif)
                # calculatedNewWeights = self.weights[i] - subtractCalc
                #
                # self.weights[
                #     i] = calculatedNewWeights  # self.weights[i] - derivatives.T * learning_rate + (momentum * dif)

                # self.old_weights[i] = current_weights

                # dif = self.weights[i] - self.old_weights[i]
                # a = self.weights[i]
                # b = self.old_weights[i]
                #print("alsihdkaushd")
                # aoishpdiuas
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

    def train(self, input_data, epochs, shuffle, learning_rate, jump, given_error, momentum, is_last_column_class):
        mistakes = []
        x_values = []
        avg_error = None
        reached_given_error = False
        num_of_outputs = len(self.activations[-1])
        if is_last_column_class:
            combined = []
            for row in input_data:
                left = row[:-1]
                one_hot = np.zeros(num_of_outputs)
                g = int(row[self.input_num])
                one_hot[g] = 1
                right = one_hot
                combined.append(np.hstack((left, right)))
            input_data = combined
        for i in range(epochs):
            if shuffle:
                np.random.shuffle(input_data)
            sum_errors = 0
            for sample in input_data:

                self.forward(sample[:self.input_num])
                target = sample[self.input_num:]
                error = self.activations[-1] - target
                self.back_propagation(error)

                self.update_weights(learning_rate, momentum)

                if self.bias:
                    self.update_biases(learning_rate)

                sum_errors += mean_squared_error(target, self.activations[-1])
            avg_error = sum_errors / len(input_data)
            # epoke konczaca uczenie zapisujemy zawsze, zeby wykres siegal do konca
            if i % jump == 0 or avg_error <= given_error:
                stats_logger.info(f"{avg_error}")
                mistakes.append(avg_error)
                x_values.append(i)
            if avg_error <= given_error:
                print(avg_error)
                stats_logger.info(f"Uzyskano zadany poziom bledu w epoce {i}\n")
                stats_logger.info(f"blad wynosi: {avg_error}\n")
                reached_given_error = True
                break
        if not reached_given_error:
            stats_logger.info(f"Nie osiagnieto podanego bledu {given_error} po {epochs} epokach, "
                              f"koncowy blad: {avg_error}")
        # jeden wykres, rysowany z faktycznie zebranych punktow - przy przerwaniu
        # petli x_values jest krotsze niz arange(0, epochs, jump) i plot wysypywal sie
        # na "x and y must have same first dimension"
        # nowa figura na kazdy trening - bez tego matplotlib dorysowywal kolejne
        # przebiegi na te same osie i wykresy z roznych uczen sie nakladaly
        plt.figure()
        plt.plot(x_values, mistakes)
        plt.ylim(0, 0.3)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Zalezność bledu od ilości epok")
        plt.xlabel("Liczba epok")
        plt.ylabel("Wartosc bledu")
        plt.show()

    def test(self, test_data, is_last_column_class):
        stats_logger.info("\n\n\nTestowanie\n\n")
        nr_of_outputs = self.neurons_in_particular_layers[-1]

        if is_last_column_class:
            combined = []
            for row in test_data:
                left = row[:-1]
                one_hot = np.zeros(nr_of_outputs)
                g = int(row[self.input_num])
                one_hot[g] = 1
                right = one_hot
                combined.append(np.hstack((left, right)))
            test_data = combined

        correctly_guessed = 0
        incorrectly_guessed = 0
        confusion_matrix = np.zeros((nr_of_outputs, nr_of_outputs))
        for i, sample in enumerate(test_data):
            target = sample[self.input_num:]
            sample_input = sample[:self.input_num]
            self.forward(sample_input)
            output = self.activations[-1]
            predicted_class = np.argmax(output)
            target_class = int(np.argmax(target))
            error = mean_squared_error(target, self.activations[-1])
            stats_logger.info(
                f"activations: {['{:.5f}'.format(a) for a in output]}, target: {target}, error: {'{:.5f}'.format(error)}")
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
            if (tp + fp != 0):
                precision = tp / (tp + fp)
            else:
                precision = 0
            if (tp != 0):
                recall = tp / (tp + fn)
            else:
                recall = 0
            if (precision + recall != 0):
                f_measure = 2 * (precision * recall) / (precision + recall)
            else:
                f_measure = 0
            confusion_string += f"Precision: {precision} Recall: {recall} F-measure: {f_measure} \n\n"
            # stats_logger.info(confusion_string)
            correctly_guessed += tp
            incorrectly_guessed += fp
        print(
            f"Dobrze zakwalifikowano {round((correctly_guessed / (correctly_guessed + incorrectly_guessed) * 100), 5)} %")
        print(confusion_matrix)
        effects_logger.info(confusion_string)
        effects_logger.info(f"\n correctly classified: {correctly_guessed}\n")
        effects_logger.info(f"\n incorrectly classified: {incorrectly_guessed}\n")
