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

    def __init__(self, initial_weights, weights, errors, epochs, converged, learning_rate,
                 stop_reason="limit epok"):
        self.initial_weights = initial_weights
        self.weights = weights
        self.errors = errors
        self.epochs = epochs
        self.converged = converged
        self.learning_rate = learning_rate
        self.stop_reason = stop_reason

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

    def train(self, X, t, learning_rate, max_epochs=10000, target_error=1e-12,
              patience=200, stall_tolerance=1e-12):
        """Algorytm 3.2 - przyrostowa regula delty na wielu wzorcach.

        Dla kazdej epoki przechodzi po wszystkich M wzorcach i po kazdym z nich
        od razu poprawia wagi:

            y_j = w . x_j
            w <- w + eta * (t_j - y_j) * x_j

        Uczenie konczy sie, gdy blad sredniokwadratowy spadnie do target_error,
        gdy przestanie sie poprawiac przez patience epok, albo po max_epochs
        epokach. Wykrywanie stagnacji jest istotne dla przypadku N < M: tam uklad
        nie ma rozwiazania dokladnego, wiec blad nigdy nie osiagnie progu i bez
        tego warunku kazdy przebieg mielilby pelny limit epok bez zadnego zysku.

        Wazna wlasnosc, z ktorej korzystaja pozniej eksperymenty: kazda poprawka
        jest wielokrotnoscia pewnego x_j, wiec caly przyrost wag w - w0 lezy w
        przestrzeni wierszowej macierzy X. Skladowa wag lezaca w jadrze X nigdy
        nie jest ruszana i zostaje taka, jaka byla w losowaniu startowym.
        """
        X = np.asarray(X, dtype=float)
        t = np.asarray(t, dtype=float)
        if X.shape[1] != self.weight_num:
            raise ValueError(
                f"Neuron ma {self.weight_num} wag, a wzorce maja {X.shape[1]} skladowych.")
        if len(X) != len(t):
            raise ValueError(f"{len(X)} wzorcow, ale {len(t)} wartosci oczekiwanych.")

        initial_weights = self.weights.copy()
        errors = []
        converged = False
        stop_reason = "limit epok"
        order = np.arange(len(X))
        best_error = np.inf
        epochs_without_progress = 0
        epoch = 0
        for epoch in range(1, max_epochs + 1):
            for j in order:
                x_j = X[j]
                error_j = t[j] - self.forward(x_j)
                self.weights = self.weights + learning_rate * error_j * x_j
            current = self.mse(X, t)
            errors.append(current)

            if not np.isfinite(current):
                stop_reason = "rozbieznosc - zbyt duzy wspolczynnik uczenia"
                break
            if current <= target_error:
                converged = True
                stop_reason = "osiagnieto zadany blad"
                break
            if current < best_error * (1.0 - stall_tolerance):
                best_error = current
                epochs_without_progress = 0
            else:
                best_error = min(best_error, current)
                epochs_without_progress += 1
                if epochs_without_progress >= patience:
                    stop_reason = f"blad przestal spadac przez {patience} epok"
                    break
        return TrainingResult(initial_weights, self.weights.copy(),
                              np.array(errors), epoch, converged, learning_rate,
                              stop_reason)


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
