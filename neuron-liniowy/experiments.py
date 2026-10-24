"""Eksperymenty z trenowaniem neuronu liniowego dla N < M, N = M i N > M.

Dla kazdego przypadku zbior treningowy jest losowany RAZ i uzywany przez
wszystkie przebiegi, a wagi startowe sa losowane osobno dla kazdego przebiegu -
dokladnie tak, jak wymaga tresc zadania. Dzieki temu jedyna roznica miedzy
przebiegami w obrebie jednego przypadku sa wagi poczatkowe.
"""

import argparse
import os

import numpy as np

from linear_neuron import (LinearNeuron, expected_delta_rule_limit,
                           least_squares_solution, safe_learning_rate)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wyniki")

CASES = [
    ("N < M", 3, 10, "uklad nadokreslony - wiecej rownan niz niewiadomych"),
    ("N = M", 5, 5, "uklad kwadratowy - tyle samo rownan co niewiadomych"),
    ("N > M", 10, 4, "uklad niedookreslony - mniej rownan niz niewiadomych"),
]


class CaseResult:
    """Komplet wynikow jednego przypadku (N vs M) ze wszystkich przebiegow."""

    def __init__(self, label, N, M, description, X, t, learning_rate, runs):
        self.label = label
        self.N = N
        self.M = M
        self.description = description
        self.X = X
        self.t = t
        self.learning_rate = learning_rate
        self.runs = runs

        self.final_weights = np.array([r.weights for r in runs])
        self.final_errors = np.array([r.final_error for r in runs])
        self.rank = int(np.linalg.matrix_rank(X))
        self.least_squares = least_squares_solution(X, t)
        self.least_squares_error = float(np.mean((X @ self.least_squares - t) ** 2))

    @property
    def weight_spread(self):
        """Najwieksza roznica tej samej wagi miedzy dowolnymi dwoma przebiegami."""
        return float((self.final_weights.max(axis=0) - self.final_weights.min(axis=0)).max())

    @property
    def distance_to_least_squares(self):
        return float(np.abs(self.final_weights - self.least_squares).max())

    @property
    def limit_prediction_error(self):
        """Odchylka od przewidywanej granicy w0 + X^+ (t - X w0) dla kazdego przebiegu."""
        return float(max(
            np.abs(r.weights - expected_delta_rule_limit(self.X, self.t, r.initial_weights)).max()
            for r in self.runs))

    @property
    def difference_in_null_space(self):
        """||X (w_i - w_1)||. Zero oznacza, ze przebiegi roznia sie tylko w jadrze X."""
        diffs = self.final_weights - self.final_weights[0]
        return float(np.abs(self.X @ diffs.T).max())

    @property
    def effective_training(self):
        """Czy trening byl skuteczny, czyli czy neuron odtwarza wszystkie wzorce."""
        return bool(self.final_errors.max() < 1e-8)


def run_case(label, N, M, description, runs=8, max_epochs=200000,
             target_error=1e-14, data_seed=2024, margin=0.1):
    """Losuje jeden zbior treningowy i trenuje na nim neuron wiele razy."""
    data_rng = np.random.default_rng(data_seed)
    X = data_rng.normal(0.0, 1.0, (M, N))
    t = data_rng.normal(0.0, 1.0, M)
    learning_rate = safe_learning_rate(X, margin)

    results = []
    for run in range(runs):
        neuron = LinearNeuron(N, np.random.default_rng(data_seed + 1000 * (run + 1)))
        results.append(neuron.train(X, t, learning_rate,
                                    max_epochs=max_epochs, target_error=target_error))
    return CaseResult(label, N, M, description, X, t, learning_rate, results)


# ---------------------------------------------------------------- raport

def case_report(case):
    """Tekstowy raport z jednego przypadku - trafia i na ekran, i do sprawozdania."""
    lines = []
    lines.append(f"### Przypadek {case.label}  (N = {case.N} wag, M = {case.M} wzorcow)")
    lines.append("")
    lines.append(f"{case.description}. Rzad macierzy X: {case.rank} "
                 f"(X ma wymiar {case.M} x {case.N}). Wspolczynnik uczenia: "
                 f"{case.learning_rate:.5g}.")
    lines.append("")
    lines.append("Wagi koncowe w kolejnych przebiegach (kazdy start z innych wag losowych):")
    lines.append("")
    header = "| przebieg | " + " | ".join(f"w{i + 1}" for i in range(case.N)) + \
             " | blad MSE | epoki |"
    lines.append(header)
    lines.append("|" + "---|" * (case.N + 3))
    for i, run in enumerate(case.runs, start=1):
        weights = " | ".join(f"{v:.4f}" for v in run.weights)
        lines.append(f"| {i} | {weights} | {run.final_error:.3e} | {run.epochs} |")
    lines.append("")
    lines.append(f"- Rozrzut wag miedzy przebiegami (max-min na wage): "
                 f"**{case.weight_spread:.3e}**")
    lines.append(f"- Blad koncowy: od {case.final_errors.min():.3e} "
                 f"do {case.final_errors.max():.3e}")
    lines.append(f"- Blad rozwiazania najmniejszych kwadratow (dolna granica): "
                 f"{case.least_squares_error:.6f}")
    lines.append(f"- Odleglosc wag koncowych od pseudoodwrotnosci X^+ t: "
                 f"{case.distance_to_least_squares:.3e}")
    lines.append(f"- Odchylka od przewidywanej granicy w0 + X^+(t - X w0): "
                 f"{case.limit_prediction_error:.3e}")
    lines.append(f"- ||X (w_i - w_1)||, czyli czy roznice leza w jadrze X: "
                 f"{case.difference_in_null_space:.3e}")
    lines.append(f"- Powod zatrzymania treningu: {case.runs[0].stop_reason}")
    lines.append(f"- Skuteczny trening (neuron odtwarza wszystkie wzorce): "
                 f"**{'TAK' if case.effective_training else 'NIE'}**")
    lines.append("")
    return "\n".join(lines)


def make_plots(cases, output_dir):
    """Rysuje zbieznosc bledu i rozrzut wag. Bez matplotlib po prostu pomija."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib niedostepny - pomijam wykresy, liczby i tak sa policzone)")
        return []

    os.makedirs(output_dir, exist_ok=True)
    written = []

    figure, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4))
    for axis, case in zip(np.atleast_1d(axes), cases):
        for i, run in enumerate(case.runs, start=1):
            axis.semilogy(np.arange(1, len(run.errors) + 1), run.errors,
                          linewidth=1, label=f"przebieg {i}")
        if case.least_squares_error > 1e-12:
            axis.axhline(case.least_squares_error, color="red", linestyle="--",
                         linewidth=1.2, label="granica najmn. kwadratow")
        axis.set_title(f"{case.label}   (N={case.N}, M={case.M})")
        axis.set_xlabel("epoka")
        axis.set_ylabel("blad MSE")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Zbieznosc bledu neuronu liniowego")
    figure.tight_layout()
    path = os.path.join(output_dir, "zbieznosc_bledu.png")
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)

    figure, axes = plt.subplots(1, len(cases), figsize=(5 * len(cases), 4))
    for axis, case in zip(np.atleast_1d(axes), cases):
        positions = np.arange(1, case.N + 1)
        for i, run in enumerate(case.runs, start=1):
            axis.plot(positions, run.weights, marker="o", markersize=4,
                      linewidth=1, alpha=0.75, label=f"przebieg {i}")
        axis.plot(positions, case.least_squares, color="black", marker="s",
                  markersize=5, linewidth=2, linestyle="--", label="X^+ t")
        axis.set_title(f"{case.label}   rozrzut {case.weight_spread:.2e}")
        axis.set_xlabel("numer wagi")
        axis.set_ylabel("wartosc wagi koncowej")
        axis.set_xticks(positions)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Wagi koncowe w kolejnych przebiegach")
    figure.tight_layout()
    path = os.path.join(output_dir, "wagi_koncowe.png")
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)
    return written


def eta_bias_study(case, factors=(1, 0.5, 0.25, 0.125), max_epochs=400000):
    """Bada, jak wspolczynnik uczenia wplywa na odleglosc od rozwiazania dokladnego.

    Regula przyrostowa poprawia wagi po kazdym wzorcu z osobna, wiec przy stalym
    eta nie zatrzymuje sie w punkcie najmniejszych kwadratow, tylko w jego
    otoczeniu o promieniu rzedu eta.
    """
    rows = []
    for factor in factors:
        learning_rate = case.learning_rate * factor
        neuron = LinearNeuron(case.N, np.random.default_rng(7))
        result = neuron.train(case.X, case.t, learning_rate,
                              max_epochs=max_epochs, target_error=1e-14)
        gap = float(np.abs(result.weights - case.least_squares).max())
        rows.append((learning_rate, gap, result.final_error))
    return rows


def robustness_study(seeds=(2024, 7, 99, 12345, 555), runs=6):
    """Powtarza wszystkie trzy przypadki dla kilku losowan zbioru treningowego.

    Jeden wylosowany zbior to za malo, zeby mowic o prawidlowosci - bez tego
    sprawdzenia nie wiadomo, czy obserwacje nie sa artefaktem jednego losowania.
    """
    rows = []
    for seed in seeds:
        entry = {"seed": seed}
        for label, N, M, description in CASES:
            case = run_case(label, N, M, description, runs=runs, data_seed=seed)
            entry[label] = (case.weight_spread, float(case.final_errors.max()))
        rows.append(entry)
    return rows


def rank_deficient_study(runs=6, seed=1):
    """N = M, ale macierz X jest osobliwa - sprawdza, co naprawde rzadzi wynikiem.

    Losowe macierze gaussowskie sa prawie na pewno pelnego rzedu, wiec w glownych
    eksperymentach przypadek N = M zawsze wychodzi jednoznaczny. Tutaj wiersze X sa
    celowo liniowo zalezne. Jesli decyduje rzad X, a nie samo porownanie N z M, to
    ten przypadek powinien zachowac sie jak uklad niedookreslony.
    """
    rng = np.random.default_rng(seed)
    N = 5
    base = rng.normal(0.0, 1.0, (3, N))
    X = np.vstack([base, base[0] + base[1], base[1] - base[2]])
    t = X @ rng.normal(0.0, 1.0, N)          # t zgodne z X, wiec rozwiazanie istnieje
    learning_rate = safe_learning_rate(X)

    final_weights = []
    for run in range(runs):
        neuron = LinearNeuron(N, np.random.default_rng(900 + run))
        final_weights.append(
            neuron.train(X, t, learning_rate, max_epochs=200000,
                         target_error=1e-14).weights)
    final_weights = np.array(final_weights)
    rank = int(np.linalg.matrix_rank(X))
    return {
        "N": N,
        "M": len(X),
        "rank": rank,
        "null_dim": N - rank,
        "spread": float((final_weights.max(axis=0) - final_weights.min(axis=0)).max()),
        "error": float(np.mean((X @ final_weights[0] - t) ** 2)),
        "null_check": float(np.abs(X @ (final_weights - final_weights[0]).T).max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Eksperymenty z neuronem liniowym dla N<M, N=M i N>M.")
    parser.add_argument("--runs", type=int, default=8,
                        help="liczba przebiegow treningu w kazdym przypadku")
    parser.add_argument("--max-epochs", type=int, default=200000)
    parser.add_argument("--data-seed", type=int, default=2024,
                        help="ziarno losowania zbioru treningowego")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    arguments = parser.parse_args()

    cases = []
    for label, N, M, description in CASES:
        print(f"Licze przypadek {label} (N={N}, M={M})...")
        cases.append(run_case(label, N, M, description, runs=arguments.runs,
                              max_epochs=arguments.max_epochs,
                              data_seed=arguments.data_seed))

    sections = [case_report(case) for case in cases]
    report = "\n".join(sections)
    print("\n" + report)

    overdetermined = cases[0]
    bias_rows = eta_bias_study(overdetermined)
    bias_lines = ["### Wplyw wspolczynnika uczenia na dokladnosc (przypadek N < M)", "",
                  "| eta | odleglosc od X^+ t | blad MSE |", "|---|---|---|"]
    for learning_rate, gap, error in bias_rows:
        bias_lines.append(f"| {learning_rate:.5g} | {gap:.3e} | {error:.6f} |")
    bias_lines.append("")
    bias_text = "\n".join(bias_lines)
    print(bias_text)

    print("Licze sprawdzenie powtarzalnosci dla kilku losowan zbioru...")
    robustness = robustness_study()
    robust_lines = ["### Powtarzalnosc dla roznych losowan zbioru treningowego", "",
                    "| ziarno | " + " | ".join(f"{label} rozrzut / blad" for label, _, _, _ in CASES) + " |",
                    "|---|" + "---|" * len(CASES)]
    for entry in robustness:
        cells = " | ".join(f"{entry[label][0]:.2e} / {entry[label][1]:.2e}"
                           for label, _, _, _ in CASES)
        robust_lines.append(f"| {entry['seed']} | {cells} |")
    robust_lines.append("")
    robust_text = "\n".join(robust_lines)
    print(robust_text)

    deficient = rank_deficient_study()
    deficient_text = "\n".join([
        "### Co naprawde rzadzi wynikiem: N vs M czy rzad macierzy X?", "",
        f"Macierz X o wymiarze {deficient['M']} x {deficient['N']} (czyli N = M), "
        f"ale o liniowo zaleznych wierszach: rzad = {deficient['rank']}, "
        f"wymiar jadra = {deficient['null_dim']}.", "",
        f"- rozrzut wag miedzy przebiegami: **{deficient['spread']:.3e}**",
        f"- blad koncowy: {deficient['error']:.3e}",
        f"- ||X (w_i - w_1)||: {deficient['null_check']:.3e}", "",
        "Mimo N = M wagi koncowe sa rozne w kazdym przebiegu, a blad zerowy - czyli "
        "zachowanie przypadku niedookreslonego. Decyduje **rzad macierzy X**, a nie samo "
        "porownanie N z M. Relacja N vs M jest tylko typowym nastepstwem rzedu, bo losowe "
        "macierze sa prawie na pewno pelnego rzedu.", ""])
    print(deficient_text)

    plots = make_plots(cases, arguments.output_dir)
    os.makedirs(arguments.output_dir, exist_ok=True)
    results_path = os.path.join(arguments.output_dir, "wyniki.md")
    with open(results_path, "w", encoding="utf-8") as handle:
        handle.write("# Wyniki eksperymentow\n\n")
        handle.write(f"Liczba przebiegow na przypadek: {arguments.runs}. "
                     f"Ziarno zbioru treningowego: {arguments.data_seed}.\n\n")
        handle.write(report + "\n" + bias_text + "\n" + robust_text
                     + "\n" + deficient_text + "\n")
    print(f"\nZapisano wyniki do {results_path}")
    for path in plots:
        print(f"Zapisano wykres {path}")


if __name__ == "__main__":
    main()
