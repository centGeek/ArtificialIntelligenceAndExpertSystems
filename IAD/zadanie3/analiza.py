"""Analiza treningu perceptronu 4-2-4 - tabele i wykresy do sprawozdania.

Uruchomienie:  python analiza.py
Wyniki trafiaja do katalogu wyniki/.
"""

import os
import sys

import numpy as np

from mlp import (IDENTITY_PATTERNS as WZORCE, classify, error, hidden,
                 make_perceptron, outputs, train)

KATALOG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(KATALOG, "wyniki")
EPOKI = 20000
KROK = 0.5
PRZEBIEGI = 10


def trening_wielokrotny(z_biasem, przebiegi=PRZEBIEGI, krok=KROK, epoki=EPOKI):
    """Zwraca liste (siec, krzywa bledu) dla kolejnych wag startowych."""
    out = []
    for ziarno in range(przebiegi):
        siec = make_perceptron(z_biasem, np.random.default_rng(ziarno))
        out.append((siec, train(siec, WZORCE, krok, epoki)))
    return out


def granice_decyzyjne(siec):
    """Wspolczynniki (u1, u2, b) granic decyzyjnych neuronow wyjsciowych."""
    warstwa = siec.layer_list[1]
    return [(float(n.weights[0]), float(n.weights[1]),
             float(n.bias) if warstwa.bias else 0.0) for n in warstwa.neurons]


def ile_da_sie_odciac(punkty, przez_zero):
    """Ile z 4 punktow da sie oddzielic prosta od pozostalych trzech.

    przez_zero=True odpowiada neuronowi BEZ biasu: granica u.h = 0 musi
    przechodzic przez poczatek ukladu. False to neuron Z biasem, gdzie granica
    u.h + b = 0 moze lezec gdziekolwiek.
    """
    ile = 0
    for k in range(4):
        reszta = [punkty[j] for j in range(4) if j != k]
        if przez_zero:
            kierunki = [np.array([np.cos(t), np.sin(t)])
                        for t in np.linspace(0, 2*np.pi, 20000, endpoint=False)]
            ok = any((punkty[k] @ u) > 1e-9 and all((o @ u) < -1e-9 for o in reszta)
                     for u in kierunki)
        else:
            # punkt da sie odciac dowolna prosta, gdy lezy poza otoczka wypukla
            # pozostalych trzech
            siatka = np.linspace(0, 1, 120)
            ok = min(np.linalg.norm(a*reszta[0] + b*reszta[1]
                                    + (1-a-b)*reszta[2] - punkty[k])
                     for a in siatka for b in siatka if a + b <= 1) > 1e-2
        ile += ok
    return ile


def wykres_zbieznosci(z_bias, bez_bias):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib niedostepny - pomijam wykres zbieznosci)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (_, krzywa), kolor, opis in ((min(z_bias, key=lambda t: t[1][-1]), "C0", "z BIAS"),
                                     (min(bez_bias, key=lambda t: t[1][-1]), "C3", "bez BIAS")):
        ax.loglog(np.arange(1, len(krzywa)+1), krzywa, color=kolor, lw=1.4, label=opis)
    ax.set_xlabel("epoka"); ax.set_ylabel("blad E(w)")
    ax.set_title("Zbieznosc bledu, najlepszy przebieg z 10")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "zbieznosc.png"), dpi=130)
    plt.close(fig)
    print("Zapisano wykres zbieznosc.png")


def wykres_geometrii(z_bias, bez_bias):
    """Polozenia wzorcow w warstwie ukrytej wraz z granicami decyzyjnymi."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib niedostepny - pomijam wykres geometrii)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
    for ax, (nazwa, zbior) in zip(axes, (("z BIAS", z_bias), ("bez BIAS", bez_bias))):
        siec = min(zbior, key=lambda t: t[1][-1])[0]
        P = np.array([hidden(siec, x) for x, _ in WZORCE])
        zasieg = max(np.abs(P).max() * 1.6, 1.0)
        for i, (u1, u2, b) in enumerate(granice_decyzyjne(siec)):
            xs = np.array([-zasieg, zasieg])
            if abs(u2) > 1e-9:
                ax.plot(xs, -(u1*xs + b)/u2, color=f"C{i}", lw=1, alpha=0.7)
            elif abs(u1) > 1e-9:
                ax.axvline(-b/u1, color=f"C{i}", lw=1, alpha=0.7)
        for i, p in enumerate(P):
            ax.scatter(*p, s=90, color=f"C{i}", zorder=3, edgecolor="k", linewidth=0.5)
            ax.annotate(f"{i+1}", p, textcoords="offset points", xytext=(7, 5), fontsize=9)
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlim(-zasieg, zasieg); ax.set_ylim(-zasieg, zasieg)
        ax.set_xlabel("$h_1$"); ax.set_ylabel("$h_2$")
        ax.set_title(f"{nazwa}: odcinalne {ile_da_sie_odciac(P, nazwa=='bez BIAS')} z 4",
                     fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle("Wzorce w warstwie ukrytej i granice decyzyjne neuronow wyjsciowych")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "geometria.png"), dpi=130)
    plt.close(fig)
    print("Zapisano wykres geometria.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    z_bias = trening_wielokrotny(True)
    bez_bias = trening_wielokrotny(False)

    t = "# Wyniki eksperymentow\n\n"
    t += f"Parametry: krok uczacy {KROK}, {EPOKI} epok, {PRZEBIEGI} przebiegow.\n\n"

    t += "### Blad koncowy po treningu\n\n"
    t += "| wariant | min | mediana | max | poprawnie sklasyfikowanych |\n|---|---|---|---|---|\n"
    for nazwa, zbior in (("z BIAS", z_bias), ("bez BIAS", bez_bias)):
        b = np.array([k[-1] for _, k in zbior])
        naj = min(zbior, key=lambda x: x[1][-1])[0]
        t += (f"| {nazwa} | {b.min():.6f} | {np.median(b):.6f} | {b.max():.6f} | "
              f"{classify(naj, WZORCE)}/4 |\n")

    for nazwa, zbior in (("z BIAS", z_bias), ("bez BIAS", bez_bias)):
        naj = min(zbior, key=lambda x: x[1][-1])[0]
        t += f"\n### Wyjscia sieci {nazwa} (najlepszy przebieg)\n\n"
        t += "| wzorzec | y1 | y2 | y3 | y4 | najwieksze |\n|---|---|---|---|---|---|\n"
        for i, (x, _) in enumerate(WZORCE, start=1):
            y = outputs(naj, x)
            t += (f"| {i} | " + " | ".join(f"{v:.3f}" for v in y)
                  + f" | {int(np.argmax(y))+1} |\n")
        t += f"\n### Polozenia wzorcow w warstwie ukrytej {nazwa}\n\n"
        t += "| wzorzec | h1 | h2 |\n|---|---|---|\n"
        P = np.array([hidden(naj, x) for x, _ in WZORCE])
        for i, p in enumerate(P, start=1):
            t += f"| {i} | {p[0]:.3f} | {p[1]:.3f} |\n"
        t += (f"\n- prosta dowolna (neuron z biasem): **{ile_da_sie_odciac(P, False)} z 4**\n"
              f"- prosta przez poczatek (neuron bez biasu): **{ile_da_sie_odciac(P, True)} z 4**\n")

    t += "\n### Czy blad bez biasu to granica, czy wolna zbieznosc?\n\n"
    t += "| krok uczacy | epoki | najlepszy blad |\n|---|---|---|\n"
    for krok in (0.1, 0.5, 2.0):
        for epoki in (20000, 50000):
            naj = min(k[-1] for _, k in trening_wielokrotny(False, 5, krok, epoki))
            t += f"| {krok} | {epoki} | {naj:.6f} |\n"

    print(t)
    with open(os.path.join(OUT, "wyniki.md"), "w", encoding="utf-8") as f:
        f.write(t)
    print(f"Zapisano wyniki do {OUT}/wyniki.md")
    wykres_zbieznosci(z_bias, bez_bias)
    wykres_geometrii(z_bias, bez_bias)
