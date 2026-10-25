"""Eksperymenty z trenowaniem neuronu liniowego dla N < M, N = M i N > M.

Zbior treningowy jest losowany raz na przypadek i uzywany przez wszystkie
przebiegi, a wagi startowe losowane osobno dla kazdego przebiegu - dokladnie
tak, jak wymaga tresc zadania.
"""

import os
import sys

import numpy as np

from linear_neuron import (expected_delta_rule_limit, least_squares_solution,
                           make_neuron, train, weights)

EPOCHS = 20000          # K z Algorytmu 3.2 - jedyne kryterium stopu
LEARNING_RATE = 0.1     # eta dobrane eksperymentalnie, wspolne dla wszystkich przypadkow
RUNS = 8
SEED = 2024
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wyniki")


def progress(text):
    """Postep idzie na stderr, zeby przekierowanie wynikow do pliku go nie polknelo."""
    sys.stderr.write(text)
    sys.stderr.flush()


CASES = [("N < M", 3, 10, "uklad nadokreslony - wiecej rownan niz niewiadomych"),
         ("N = M", 5, 5, "uklad kwadratowy - tyle samo rownan co niewiadomych"),
         ("N > M", 10, 4, "uklad niedookreslony - mniej rownan niz niewiadomych")]


def run_case(N, M, runs=RUNS, epochs=EPOCHS, seed=SEED):
    """Trenuje neuron wielokrotnie na jednym zbiorze i liczy miary porownawcze."""
    rng = np.random.default_rng(seed)
    X, t = rng.normal(0, 1, (M, N)), rng.normal(0, 1, M)
    eta = LEARNING_RATE
    progress(f"  N={N:<3} M={M:<3} ziarno={seed:<6} {runs} x {epochs} epok ")
    results = []
    for i in range(runs):
        neuron = make_neuron(N, np.random.default_rng(seed + 1000 * (i + 1)))
        initial, errors = train(neuron, X, t, eta, epochs)
        results.append((weights(neuron).copy(), initial, errors))
        progress(".")
    progress(" gotowe\n")
    W = np.array([w for w, _, _ in results])
    ls = least_squares_solution(X, t)
    return dict(
        X=X, t=t, eta=eta, results=results, W=W, ls=ls,
        rank=int(np.linalg.matrix_rank(X)),
        spread=float((W.max(axis=0) - W.min(axis=0)).max()),
        error=float(max(e[-1] for _, _, e in results)),
        ls_error=float(np.mean((X @ ls - t) ** 2)),
        to_ls=float(np.abs(W - ls).max()),
        # czy kazdy przebieg trafil w przewidywana granice w0 + X+(t - X w0)
        to_limit=float(max(np.abs(w - expected_delta_rule_limit(X, t, w0)).max()
                           for w, w0, _ in results)),
        # zero oznacza, ze przebiegi roznia sie tylko skladowa z jadra X
        null=float(np.abs(X @ (W - W[0]).T).max()))


def report(label, description, N, M, c):
    lines = [f"### Przypadek {label}  (N = {N} wag, M = {M} wzorcow)", "",
             f"{description}. Rzad macierzy X: {c['rank']}. "
             f"Wspolczynnik uczenia: {c['eta']:.5g}.", "",
             "| przebieg | " + " | ".join(f"w{i+1}" for i in range(N)) + " | blad MSE |",
             "|" + "---|" * (N + 2)]
    for i, (w, _, e) in enumerate(c["results"], start=1):
        lines.append(f"| {i} | " + " | ".join(f"{v:.4f}" for v in w)
                     + f" | {e[-1]:.3e} |")
    lines += ["",
              f"- rozrzut wag miedzy przebiegami: **{c['spread']:.3e}**",
              f"- blad koncowy: {c['error']:.3e}, granica najmniejszych kwadratow: {c['ls_error']:.6f}",
              f"- odleglosc od X^+ t: {c['to_ls']:.3e}",
              f"- odchylka od w0 + X^+(t - X w0): {c['to_limit']:.3e}",
              f"- ||X (w_i - w_1)||: {c['null']:.3e}",
              f"- skuteczny trening: **{'TAK' if c['error'] < 1e-8 else 'NIE'}**", ""]
    return "\n".join(lines)


def plots(cases):
    """Zbieznosc bledu i rozrzut wag. Bez matplotlib po prostu pomija."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib niedostepny - pomijam wykresy)")
        return
    for name, title, kind in [("zbieznosc_bledu", "Zbieznosc bledu neuronu liniowego", "error"),
                              ("wagi_koncowe", "Wagi koncowe w kolejnych przebiegach", "weights")]:
        # panele pionowo, nie poziomo - trzy obok siebie sa nieczytelne po
        # wstawieniu w sprawozdanie; proporcje dobrane do pola tekstu 390 pt
        fig, axes = plt.subplots(len(cases), 1, figsize=(7.5, 2.6 * len(cases)))
        for ax, (label, N, M, _), c in zip(np.atleast_1d(axes), CASES, cases):
            if kind == "error":
                for i, (_, _, e) in enumerate(c["results"], start=1):
                    ax.semilogy(np.arange(1, len(e) + 1), e, lw=1, label=f"przebieg {i}")
                if c["ls_error"] > 1e-12:
                    ax.axhline(c["ls_error"], color="red", ls="--", lw=1.2,
                               label="granica najmn. kwadratow")
                ax.set_xlabel("epoka"); ax.set_ylabel("blad MSE")
                ax.set_title(f"{label}   (N={N}, M={M})")
            else:
                pos = np.arange(1, N + 1)
                for i, (w, _, _) in enumerate(c["results"], start=1):
                    ax.plot(pos, w, marker="o", ms=4, lw=1, alpha=0.75, label=f"przebieg {i}")
                ax.plot(pos, c["ls"], color="black", marker="s", ms=5, lw=2, ls="--", label="X^+ t")
                ax.set_xlabel("numer wagi"); ax.set_ylabel("wartosc wagi"); ax.set_xticks(pos)
                ax.set_title(f"{label}   rozrzut {c['spread']:.2e}")
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(title); fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=130)
        plt.close(fig)
        print(f"Zapisano wykres {name}.png")


if __name__ == "__main__":
    progress(f"Eksperymenty: K={EPOCHS} epok, {RUNS} przebiegow na przypadek.\n"
             f"Calosc to 114 treningow, ok. 2 minut na typowej maszynie.\n\n"
             f"[1/2] trzy przypadki N vs M\n")
    cases = [run_case(N, M) for _, N, M, _ in CASES]
    text = "\n".join(report(l, d, N, M, c) for (l, N, M, d), c in zip(CASES, cases))

    progress("\n\n[2/2] powtarzalnosc dla 5 ziaren - najdluzszy etap, ok. 1.5 minuty\n")
    text += ("\n### Powtarzalnosc dla roznych losowan zbioru\n\n| ziarno | "
             + " | ".join(f"{l} rozrzut / blad" for l, _, _, _ in CASES) + " |\n"
             + "|---|" + "---|" * len(CASES) + "\n")
    for seed in (2024, 7, 99, 12345, 555):
        row = [run_case(N, M, runs=6, seed=seed) for _, N, M, _ in CASES]
        text += f"| {seed} | " + " | ".join(f"{c['spread']:.2e} / {c['error']:.2e}" for c in row) + " |\n"

        progress("\n")

    print(text)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "wyniki.md"), "w", encoding="utf-8") as f:
        f.write("# Wyniki eksperymentow\n\n" + text)
    print(f"Zapisano wyniki do {OUT}/wyniki.md")
    plots(cases)
