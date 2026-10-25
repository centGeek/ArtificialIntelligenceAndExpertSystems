"""Dwuwarstwowy perceptron 4-2-4 i Algorytm 2.1 (wsteczna propagacja bledu).

Architektura dokladnie wg punktu 2.2 materialow do Wykladu nr 2:
  - 4 neurony kopiujace w warstwie wejsciowej (tylko przepisuja wejscia dalej),
  - 2 neurony LINIOWE w warstwie ukrytej (brak funkcji aktywacji, brak biasu),
  - 4 neurony SIGMOIDALNE w warstwie wyjsciowej, z wejsciem obciazajacym (BIAS).

Liczba wag: 2*4 = 8 w warstwie ukrytej oraz 4*3 = 12 w wyjsciowej, razem 20,
co zgadza sie z zapisem w ∈ R^20 w materialach.
"""

import numpy as np


def sigmoid(s):
    """Sigmoida liczona stabilnie numerycznie.

    Naiwne 1/(1+exp(-s)) przepelnia sie dla duzych ujemnych s, co przy sieci
    bez biasu (gdzie sumy wazone uciekaja w setki) zasmieca wynik ostrzezeniami.
    """
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    positive = s >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-s[positive]))
    exp_s = np.exp(s[~positive])
    out[~positive] = exp_s / (1.0 + exp_s)
    return out


def sigmoid_derivative_from_sum(s):
    """f'(s) = f(s) * (1 - f(s)) - pochodna liczona z sumy wazonej, jak w (2.1.4)."""
    value = sigmoid(s)
    return value * (1.0 - value)


class Perceptron424:
    """Siec 4-2-4 z liniowa warstwa ukryta i sigmoidalna warstwa wyjsciowa."""

    def __init__(self, use_bias=True, rng=None, weight_range=(-1.0, 1.0)):
        self.use_bias = use_bias
        self.rng = rng if rng is not None else np.random.default_rng()
        low, high = weight_range
        # krok 1) algorytmu - wagi losowe z zadanego przedzialu
        self.hidden_weights = self.rng.uniform(low, high, (2, 4))          # w^(1)
        output_inputs = 3 if use_bias else 2                                # h1, h2 (+ bias)
        self.output_weights = self.rng.uniform(low, high, (4, output_inputs))  # w^(2)

    @property
    def weight_num(self):
        return self.hidden_weights.size + self.output_weights.size

    def hidden(self, x):
        """Krok 4) - wyjscia warstwy ukrytej. Neurony liniowe, wiec bez aktywacji."""
        return self.hidden_weights @ x

    def hidden_extended(self, x):
        """Wyjscia warstwy ukrytej uzupelnione o stale wyjscie biasu h3 = 1."""
        h = self.hidden(x)
        return np.append(h, 1.0) if self.use_bias else h

    def forward(self, x):
        """Kroki 4-6). Zwraca (h_ext, s, y)."""
        h_ext = self.hidden_extended(np.asarray(x, dtype=float))
        s = self.output_weights @ h_ext              # krok 5)
        return h_ext, s, sigmoid(s)                  # krok 6)

    def outputs(self, x):
        return self.forward(x)[2]

    def train_step(self, x, z, learning_rate):
        """Jedna prezentacja wzorca: kroki 3-9) Algorytmu 2.1."""
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        h_ext, s, y = self.forward(x)

        # krok 7) blad warstwy wyjsciowej
        delta_output = sigmoid_derivative_from_sum(s) * (z - y)

        # krok 8) propagacja wsteczna do warstwy ukrytej.
        # Warstwa ukryta jest liniowa, wiec nie ma czynnika f'(.) - jego pochodna
        # jest rowna 1. Bierzemy tylko kolumny odpowiadajace neuronom ukrytym;
        # kolumna biasu nie prowadzi do zadnego neuronu, wiec nic tam nie wraca.
        delta_hidden = self.output_weights[:, :2].T @ delta_output

        # krok 9) korekta wag
        self.hidden_weights += learning_rate * np.outer(delta_hidden, x)     # (2.1.6a)
        self.output_weights += learning_rate * np.outer(delta_output, h_ext)  # (2.1.6b)
        return y

    def error(self, patterns):
        """Funkcja bledu E(w) = 1/2 * suma po wzorcach i wyjsciach (z - y)^2."""
        total = 0.0
        for x, z in patterns:
            total += np.sum((np.asarray(z, dtype=float) - self.outputs(x)) ** 2)
        return 0.5 * total

    def train(self, patterns, learning_rate, epochs):
        """Algorytm 2.1 rozszerzony na wiele wzorcow treningowych.

        Algorytm 2.1 zapisany jest dla pojedynczego wzorca, a zadanie wymaga
        czterech. Materialy stwierdzaja, ze rozszerzenie jest bezposrednie,
        wiec w kazdej epoce prezentowane sa kolejno wszystkie wzorce, a wagi
        poprawiane sa po kazdym z nich - tak samo jak w Algorytmie 3.2 dla
        neuronu liniowego.

        Jedynym kryterium stopu jest liczba epok L, zgodnie z krokiem 2).
        """
        errors = np.empty(epochs)
        for epoch in range(epochs):
            for x, z in patterns:
                self.train_step(x, z, learning_rate)
            errors[epoch] = self.error(patterns)
        return errors

    def classify(self, patterns):
        """Sprawdza, czy siec odtwarza kazdy wzorzec - najwieksze wyjscie we wlasciwym miejscu."""
        correct = 0
        for x, z in patterns:
            if int(np.argmax(self.outputs(x))) == int(np.argmax(z)):
                correct += 1
        return correct


IDENTITY_PATTERNS = [
    (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
    (np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])),
    (np.array([0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])),
    (np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])),
]
