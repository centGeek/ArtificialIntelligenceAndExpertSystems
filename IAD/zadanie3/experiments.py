"""Eksperymenty z perceptronem 4-2-4: trening z wejsciem obciazajacym i bez niego."""

import numpy as np

from mlp import IDENTITY_PATTERNS as PATTERNS, Perceptron424

EPOCHS = 20000
LEARNING_RATE = 0.5
RUNS = 10


def train_many(use_bias, runs=RUNS, learning_rate=LEARNING_RATE, epochs=EPOCHS):
    nets = []
    for seed in range(runs):
        net = Perceptron424(use_bias=use_bias, rng=np.random.default_rng(seed))
        nets.append((net, net.train(PATTERNS, learning_rate, epochs)[-1]))
    return nets


def isolable(points, through_origin):
    """Ile z 4 punktow da sie odciac prosta od pozostalych trzech.

    through_origin=True odpowiada neuronowi BEZ biasu (granica u.h = 0 musi
    przechodzic przez poczatek), False - neuronowi Z biasem (granica u.h + b = 0
    moze lezec gdziekolwiek, wiec wystarczy byc wierzcholkiem otoczki wypuklej).
    """
    count = 0
    for k in range(4):
        others = [points[j] for j in range(4) if j != k]
        if through_origin:
            ok = any((points[k] @ u) > 1e-9 and all((o @ u) < -1e-9 for o in others)
                     for u in [np.array([np.cos(t), np.sin(t)])
                               for t in np.linspace(0, 2 * np.pi, 20000, endpoint=False)])
        else:
            grid = np.linspace(0, 1, 120)
            ok = min(np.linalg.norm(a * others[0] + b * others[1]
                                    + (1 - a - b) * others[2] - points[k])
                     for a in grid for b in grid if a + b <= 1) > 1e-2
        count += ok
    return count


for use_bias in (True, False):
    nets = train_many(use_bias)
    errors = np.array([e for _, e in nets])
    best = nets[int(np.argmin(errors))][0]
    points = np.array([best.hidden(x) for x, _ in PATTERNS])

    print(f"\n=== warstwa wyjsciowa {'z BIAS' if use_bias else 'bez BIAS'} ===")
    print(f"blad koncowy po {EPOCHS} epokach: min {errors.min():.6f}, "
          f"mediana {np.median(errors):.6f}, max {errors.max():.6f}")
    print("wyjscia w najlepszym przebiegu:")
    for i, (x, _) in enumerate(PATTERNS, start=1):
        y = best.outputs(x)
        print(f"   wzorzec {i}: " + "  ".join(f"{v:.3f}" for v in y)
              + f"   najwieksze na pozycji {int(np.argmax(y)) + 1}")
    print("polozenia wzorcow w warstwie ukrytej:")
    for i, p in enumerate(points, start=1):
        print(f"   wzorzec {i}: ({p[0]:8.3f}, {p[1]:8.3f})")
    print(f"da sie odciac prosta dowolna (neuron z biasem):      {isolable(points, False)} z 4")
    print(f"da sie odciac prosta przez poczatek (bez biasu):     {isolable(points, True)} z 4")

print("\n=== czy blad bez biasu to granica, czy wolna zbieznosc? ===")
for lr in (0.1, 0.5, 2.0):
    for epochs in (20000, 50000):
        best = min(e for _, e in train_many(False, 5, lr, epochs))
        print(f"   lr={lr:<4} epok={epochs:<6} najlepszy blad = {best:.6f}")
