"""Perceptron 4-2-4 i Algorytm 2.1 na bazie klasy Network z zadanie2/.

Architektura z punktu 2.2: 4 neurony kopiujace, 2 liniowe bez biasu w warstwie
ukrytej, 4 sigmoidalne z biasem na wyjsciu.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "siec"))
from NeuronNetwork import Network
from functions import (linear, linear_derivative, mean_squared_error_derivative,
                       sigmoid, sigmoid_derivative)

IDENTITY_PATTERNS = [(np.eye(4)[i], np.eye(4)[i]) for i in range(4)]


def make_perceptron(use_bias=True, rng=None, weight_range=(-1.0, 1.0)):
    """Siec 4-2-4 o architekturze z punktu 2.2."""
    net = Network([4, 2, 4], bias=[False, use_bias],
                  activations=[(linear, linear_derivative),
                               (sigmoid, sigmoid_derivative)])
    # wagi z wlasnego generatora, bo Network losuje globalnym, a eksperymenty
    # musza byc powtarzalne
    rng = rng if rng is not None else np.random.default_rng()
    low, high = weight_range
    for layer in net.layer_list:
        for neuron in layer.neurons:
            neuron.weights = rng.uniform(low, high, len(neuron.weights))
            if layer.bias:
                neuron.bias = float(rng.uniform(low, high))
    return net


def outputs(net, x):
    # bez biasu sumy wazone uciekaja w setki i exp sie przepelnia; wynik jest
    # poprawny (1/(1+inf) = 0), wiec tylko wyciszamy ostrzezenie
    with np.errstate(over="ignore"):
        net.forward(np.asarray(x, dtype=float))
    return net.activations[-1]


def hidden(net, x):
    """Wyjscia warstwy ukrytej - polozenie wzorca w przestrzeni dwuwymiarowej."""
    with np.errstate(over="ignore"):
        net.forward(np.asarray(x, dtype=float))
    return net.activations[1]


def error(net, patterns):
    """Funkcja bledu E(w) = 1/2 * suma po wzorcach i wyjsciach (z - y)^2."""
    return 0.5 * sum(float(np.sum((np.asarray(z, dtype=float) - outputs(net, x)) ** 2))
                     for x, z in patterns)


def train(net, patterns, learning_rate, epochs):
    """Algorytm 2.1 rozszerzony na cztery wzorce, dokladnie L epok."""
    errors = np.empty(epochs)
    for k in range(epochs):
        for x, z in patterns:
            with np.errstate(over="ignore"):
                net.forward(np.asarray(x, dtype=float))
            net.back_propagation(mean_squared_error_derivative(
                np.asarray(z, dtype=float), net.activations[-1]))
            # mean_squared_error_derivative zwraca 2(y-z)/len(y), wiec krok
            # Network to polowa kroku eta z Algorytmu 2.1
            net.update_weights(learning_rate * 2.0, 0)
            if any(net.layer_biases):
                net.update_biases(learning_rate * 2.0)
        errors[k] = error(net, patterns)
    return errors


def classify(net, patterns):
    """Ile wzorcow ma najwieksze wyjscie we wlasciwym miejscu."""
    return sum(int(np.argmax(outputs(net, x))) == int(np.argmax(z)) for x, z in patterns)
