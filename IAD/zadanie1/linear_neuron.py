"""Neuron liniowy i jego trening wieloma wzorcami (Algorytm 3.2 z Wykladu 1).

Neuron liniowy liczy y = w . x, bez funkcji aktywacji. Dzieki temu trening jest
rownowazny rozwiazywaniu ukladu M rownan liniowych z N niewiadomymi:

    X w = t,    X ma wymiar M x N

gdzie wierszami X sa wzorce treningowe, a t to wektor wartosci oczekiwanych.
Wagi neuronu sa niewiadomymi tego ukladu - i stad biora sie wszystkie obserwacje
z eksperymentow dla N < M, N = M i N > M.
"""

import numpy as np


class TrainingResult:
    """Wynik jednego przebiegu treningu."""

    def __init__(self, initial_weights, weights, errors, epochs, learning_rate):
        self.initial_weights = initial_weights
        self.weights = weights
        self.errors = errors
        self.epochs = epochs
        self.learning_rate = learning_rate

    @property
    def final_error(self):
        return self.errors[-1] if len(self.errors) else float("nan")


class LinearNeuron:
    """Neuron liniowy: y = w . x."""

    def __init__(self, weight_num, rng=None):
        self.weight_num = weight_num
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weights = None
        self.randomize_weights()

    def randomize_weights(self, low=-1.0, high=1.0):
        """Losuje wagi startowe. Kazdy przebieg eksperymentu startuje z innych."""
        self.weights = self.rng.uniform(low, high, self.weight_num)
        return self.weights

    def forward(self, x):
        return float(np.dot(self.weights, x))

    def outputs(self, X):
        return X @ self.weights

    def mse(self, X, t):
        return float(np.mean((self.outputs(X) - t) ** 2))

    def train(self, X, t, learning_rate, epochs):
        """Algorytm 3.2 - regula delta treningu neuronu liniowego dla wielu wzorcow.

        Wierna transkrypcja krokow 4-10 algorytmu:

            4)  dopoki k < K
            5)  mu = 1
            6)  dopoki nie zaprezentowano wszystkich M wzorcow
            7)  y = suma w_i * x_i^(mu)
            8)  w_i = w_i + eta * (z^(mu) - y) * x_i^(mu),  i = 1..N
            9)  mu = mu + 1, wroc do 6
            10) k = k + 1, wroc do 4

        Algorytm nie ma zadnego kryterium stopu poza liczba epok: wykonuje
        dokladnie K epok, nawet jesli blad dawno przestal spadac. Nie ma tu
        progu bledu ani wykrywania stagnacji - to byloby odstepstwo od
        specyfikacji.

        Blad sredniokwadratowy po kazdej epoce jest zapisywany, ale sluzy
        wylacznie do oceny wynikow i rysowania wykresow. Na przebieg treningu
        nie ma zadnego wplywu.

        Wagi sa aktualizowane po kazdym wzorcu z osobna (krok 8 wykonuje sie
        wewnatrz petli po wzorcach), a wszystkie N wag korzysta z tego samego
        y policzonego w kroku 7.

        Uwaga do kroku 6: w materialach warunek brzmi "czy mu = M", co czytane
        doslownie konczy epoke przed przetworzeniem ostatniego wzorca. Przeczy
        to wyjasnieniu podanemu w tym samym kroku ("czy wszystkie wzorce
        zostaly zaprezentowane"), wiec przyjeto intencje: w kazdej epoce
        przetwarzane sa wszystkie M wzorcow.
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        if X.shape[1] != self.weight_num:
            raise ValueError(
                f"Neuron ma {self.weight_num} wag, a wzorce maja {X.shape[1]} skladowych.")
        if len(X) != len(t):
            raise ValueError(f"{len(X)} wzorcow, ale {len(t)} wartosci oczekiwanych.")

        initial_weights = self.weights.copy()
        errors = np.empty(epochs)
        for k in range(epochs):                                  # krok 4
            for mu in range(len(X)):                             # kroki 5, 6, 9
                y = self.forward(X[mu])                          # krok 7
                self.weights = self.weights + learning_rate * (t[mu] - y) * X[mu]   # krok 8
            errors[k] = self.mse(X, t)                           # tylko pomiar
        return TrainingResult(initial_weights, self.weights.copy(),
                              errors, epochs, learning_rate)


def safe_learning_rate(X, margin=0.1):
    """Dobiera wspolczynnik uczenia, przy ktorym regula delty jest stabilna.

    Dla reguly delty warunek zbieznosci wiaze sie z najwiekszna wartoscia wlasna
    macierzy X^T X. Przy zbyt duzym eta bledy rosna wykladniczo i porownywanie
    wag koncowych traci sens, wiec eta jest liczone z danych, a nie zgadywane.
    """
    X = np.asarray(X, dtype=float)
    largest = np.linalg.eigvalsh(X.T @ X).max()
    if largest <= 0:
        return 0.01
    return float(margin * 2.0 / largest)


def least_squares_solution(X, t):
    """Rozwiazanie ukladu X w = t w sensie najmniejszych kwadratow (pseudoodwrotnosc).

    Dla N < M jest to jedyny punkt minimalizujacy blad.
    Dla N = M przy nieosobliwej X jest to dokladne rozwiazanie ukladu.
    Dla N > M jest to rozwiazanie o najmniejszej normie sposrod nieskonczenie wielu.
    """
    return np.linalg.pinv(np.asarray(X, dtype=float)) @ np.asarray(t, dtype=float)


def expected_delta_rule_limit(X, t, initial_weights):
    """Punkt, do ktorego regula delty powinna zbiec startujac z initial_weights.

    Poprawki zmieniaja tylko skladowa wag z przestrzeni wierszowej X, wiec granica
    to waga startowa poprawiona o rzut bledu na te przestrzen:

        w_inf = w0 + X^+ (t - X w0)

    Dla N >= M (uklad niedookreslony) roznica miedzy przebiegami bierze sie
    wylacznie z roznych skladowych startowych w jadrze X.
    """
    X = np.asarray(X, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    return initial_weights + np.linalg.pinv(X) @ (np.asarray(t, dtype=float) - X @ initial_weights)
