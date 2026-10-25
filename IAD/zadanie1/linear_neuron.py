"""Neuron liniowy i Algorytm 3.2 na bazie klasy Network z zadanie2/."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "siec"))
from NeuronNetwork import Network
from functions import linear, linear_derivative, mean_squared_error_derivative


def make_neuron(weight_num, rng=None):
    """Siec [N, 1] bez biasu z aktywacja liniowa - pojedynczy neuron liniowy."""
    return Network([weight_num, 1], bias=False,
                   activations=[(linear, linear_derivative)], rng=rng)


def weights(neuron):
    return neuron.layer_list[0].neurons[0].weights


def train(neuron, X, t, learning_rate, epochs):
    """Algorytm 3.2 - regula delta dla wielu wzorcow, dokladnie K epok."""
    initial = weights(neuron).copy()
    errors = np.empty(epochs)
    for k in range(epochs):
        for x, z in zip(X, t):
            neuron.forward(x)
            neuron.back_propagation(
                mean_squared_error_derivative(np.array([z]), neuron.activations[-1]))
            # mean_squared_error_derivative zwraca 2(y-z)/len(y), wiec krok
            # Network to polowa kroku eta z Algorytmu 3.2
            neuron.update_weights(learning_rate / 2.0, 0)
        errors[k] = float(np.mean((X @ weights(neuron) - t) ** 2))
    return initial, errors


def safe_learning_rate(X, margin=0.1):
    """Krok uczacy liczony z najwiekszej wartosci wlasnej X^T X - przy wiekszym
    regula delta rozbiega sie i porownywanie wag traci sens."""
    largest = np.linalg.eigvalsh(np.asarray(X, dtype=float).T @ np.asarray(X, dtype=float)).max()
    return float(margin * 2.0 / largest) if largest > 0 else 0.01


def least_squares_solution(X, t):
    """Rozwiazanie X w = t w sensie najmniejszych kwadratow."""
    return np.linalg.pinv(np.asarray(X, dtype=float)) @ np.asarray(t, dtype=float)


def expected_delta_rule_limit(X, t, initial_weights):
    """Granica reguly delta: w0 + X^+ (t - X w0). Poprawki leza w przestrzeni
    wierszowej X, wiec skladowa z jadra zostaje taka jak w losowaniu."""
    X = np.asarray(X, dtype=float)
    w0 = np.asarray(initial_weights, dtype=float)
    return w0 + np.linalg.pinv(X) @ (np.asarray(t, dtype=float) - X @ w0)
