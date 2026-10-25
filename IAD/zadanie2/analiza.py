"""Analiza wynikow sieci MADALINE - tabele i wykresy do sprawozdania.

Uruchomienie:  python analiza.py
Wyniki trafiaja do katalogu wyniki/.
"""

import os
import sys

import numpy as np
from PIL import Image

from madaline import (BINARY_THRESHOLD, MadalineNetwork, image_to_vector,
                      read_description)

KATALOG = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(KATALOG, "wyniki")
POZIOMY = [0, 10, 30, 50, 90]


def katalog_testowy(poziom):
    return os.path.join(KATALOG, "training_patterns" if poziom == 0
                        else f"test_patterns_{poziom}")


def przyklad_z_wykladu():
    """Ksztalty 4x4 liter X, Y, Z z Rozdzialu 4 i zaburzona litera X.

    Materialy podaja stopnie podobienstwa 0.943, 0.816 i 0.738 - to sprawdzenie
    weryfikuje implementacje wobec wartosci znanych z gory.
    """
    wzorce = {"litera X": [1,0,0,1, 0,1,1,0, 0,1,1,0, 1,0,0,1],
              "litera Y": [1,0,0,1, 0,1,1,0, 0,1,0,0, 1,0,0,0],
              "litera Z": [1,1,1,1, 0,0,1,0, 0,1,0,0, 1,1,1,1]}
    test = np.array([1,0,1,1, 0,1,1,0, 0,1,1,0, 1,0,0,1], dtype=float)

    sys.path.insert(0, os.path.join(KATALOG, "..", "siec"))
    from NeuronNetwork import Network
    from functions import linear, linear_derivative

    siec = Network([16, len(wzorce)], bias=False,
                   activations=[(linear, linear_derivative)])
    for neuron, wektor in zip(siec.layer_list[0].neurons, wzorce.values()):
        v = np.array(wektor, dtype=float)
        neuron.weights = v / np.linalg.norm(v)
    siec.forward(test / np.linalg.norm(test))
    return dict(zip(wzorce, (float(x) for x in siec.activations[-1])))


def piksele_tuszu():
    """Liczba czarnych pikseli kazdego wzorca i jej udzial w obrazie."""
    out = {}
    for sciezka, etykieta in read_description(katalog_testowy(0)):
        with Image.open(sciezka) as uchwyt:
            piksele = np.asarray(uchwyt.convert("L"), dtype=np.int16)
        n = int((piksele < BINARY_THRESHOLD).sum())
        out[etykieta.split(",")[0]] = (n, piksele.size)
    return out


def pewnosc_teoretyczna(n, P, p):
    """Oczekiwany cosinus miedzy wzorcem a jego zaszumiona wersja.

    Tusz przezywa z prawdopodobienstwem (1-p), a z tla dochodzi p*(P-n) nowych
    czarnych pikseli, co wydluza wektor wejsciowy i obniza cosinus.
    """
    wspolne = (1 - p) * n
    dlugosc = np.sqrt((1 - p) * n + p * (P - n))
    return wspolne / (np.sqrt(n) * dlugosc)


def rozpoznawanie():
    """Dla kazdego poziomu szumu: (etykieta, rozpoznane, pewnosc, wlasne).

    'pewnosc' to wyjscie neuronu zwycieskiego, a 'wlasne' - wyjscie neuronu
    odpowiadajacego tej samej literze. Powyzej 30% szumu te dwie wartosci sie
    rozchodza, bo wygrywa neuron innej litery, i tylko 'wlasne' wolno porownywac
    z modelem analitycznym.
    """
    siec = MadalineNetwork.from_directory(katalog_testowy(0))
    litery = [e.split(",")[0].split()[-1] for e in siec.labels]
    out = {}
    for poziom in POZIOMY:
        wiersze = []
        for sciezka, etykieta in read_description(katalog_testowy(poziom)):
            wektor, _ = image_to_vector(sciezka)
            wyjscia = siec.outputs(wektor)
            nazwa, pewnosc = siec.recognize(wektor)
            litera = etykieta.split(",")[0].split()[-1]
            wlasne = float(wyjscia[litery.index(litera)])
            wiersze.append((etykieta.split(",")[0], nazwa.split(",")[0],
                            pewnosc, wlasne))
        out[poziom] = wiersze
    return out


def wykres_wzorcow():
    """Podglad liter przy kolejnych poziomach zaszumienia."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib niedostepny - pomijam wykres wzorcow)")
        return
    litery = ["a", "b", "c"]
    fig, axes = plt.subplots(len(litery), len(POZIOMY),
                             figsize=(1.5 * len(POZIOMY), 1.5 * len(litery)))
    for i, litera in enumerate(litery):
        for j, poziom in enumerate(POZIOMY):
            ax = axes[i, j]
            with Image.open(os.path.join(katalog_testowy(poziom), f"{litera}.png")) as u:
                ax.imshow(np.asarray(u.convert("L")), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{poziom}%", fontsize=10)
            if j == 0:
                ax.set_ylabel(litera, fontsize=12, rotation=0, labelpad=10)
    fig.suptitle("Znaki testowe przy kolejnych poziomach zaszumienia")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wzorce_szum.png"), dpi=130)
    plt.close(fig)
    print("Zapisano wykres wzorce_szum.png")


def wykres_pewnosci(wyniki, tusz):
    """Pewnosc rozpoznania w funkcji szumu, z naniesiona krzywa teoretyczna."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib niedostepny - pomijam wykres pewnosci)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    siatka = np.linspace(0, 0.95, 200)
    for idx, (etykieta, (n, P)) in enumerate(tusz.items()):
        litera = etykieta.split()[-1]
        kolor = f"C{idx}"
        zmierzone = [w[3] for poz in POZIOMY for w in wyniki[poz]
                     if w[0].split()[-1] == litera]
        ax.plot([p for p in POZIOMY], zmierzone, "o-", color=kolor,
                label=f"litera {litera} - pomiar")
        ax.plot(siatka * 100, [pewnosc_teoretyczna(n, P, p) for p in siatka],
                "--", color=kolor, lw=1, label=f"litera {litera} - model")
    ax.set_xlabel("poziom zaszumienia [%]")
    ax.set_ylabel("stopien podobienstwa")
    ax.set_title("Pewnosc rozpoznania w funkcji zaszumienia")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pewnosc_szum.png"), dpi=130)
    plt.close(fig)
    print("Zapisano wykres pewnosc_szum.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tusz = piksele_tuszu()
    wyniki = rozpoznawanie()
    wyklad = przyklad_z_wykladu()

    t = "# Wyniki eksperymentow\n\n"
    t += "### Weryfikacja wobec przykladu z wykladu (ksztalty 4 x 4)\n\n"
    t += "| wzorzec oryginalny | materialy | implementacja |\n|---|---|---|\n"
    for nazwa, oczekiwane in (("litera X", 0.943), ("litera Y", 0.816), ("litera Z", 0.738)):
        t += f"| {nazwa} | {oczekiwane:.3f} | **{wyklad[nazwa]:.3f}** |\n"

    t += "\n### Udzial tuszu w obrazie\n\n"
    t += "| wzorzec | piksele tuszu | wszystkie piksele | udzial |\n|---|---|---|---|\n"
    for etykieta, (n, P) in tusz.items():
        t += f"| {etykieta} | {n} | {P} | {100*n/P:.1f}% |\n"

    t += "\n### Rozpoznawanie przy rosnacym zaszumieniu\n\n"
    t += "| poziom szumu | a -> | b -> | c -> | poprawnie |\n|---|---|---|---|---|\n"
    for poziom in POZIOMY:
        kom, dobre = [], 0
        for etykieta, nazwa, pewnosc, _ in wyniki[poziom]:
            ok = etykieta.split()[-1] == nazwa.split()[-1]
            dobre += ok
            kom.append(f"{nazwa.split()[-1]} ({pewnosc:.3f})" if ok
                       else f"**{nazwa.split()[-1]}** ({pewnosc:.3f})")
        t += f"| {poziom}% | " + " | ".join(kom) + f" | **{dobre}/3** |\n"

    t += "\n### Pomiar kontra model analityczny\n\n"
    t += ("| wzorzec | poziom szumu | wyjscie wlasnego neuronu | model |\n"
          "|---|---|---|---|\n")
    for etykieta, (n, P) in tusz.items():
        litera = etykieta.split()[-1]
        for poziom in POZIOMY:
            zm = next(w[3] for w in wyniki[poziom] if w[0].split()[-1] == litera)
            t += (f"| {etykieta} | {poziom}% | {zm:.3f} | "
                  f"{pewnosc_teoretyczna(n, P, poziom/100):.3f} |\n")

    print(t)
    with open(os.path.join(OUT, "wyniki.md"), "w", encoding="utf-8") as f:
        f.write(t)
    print(f"Zapisano wyniki do {OUT}/wyniki.md")
    wykres_wzorcow()
    wykres_pewnosci(wyniki, tusz)
