"""Siec MADALINE do rozpoznawania znakow oraz obsluga obrazow i plikow opisu.

Siec sklada sie z jednej warstwy neuronow liniowych. Kazdy neuron odpowiada za
jeden znak wzorcowy, a jego wektor wagowy to znormalizowany wektor tego znaku.
Wyjscie neuronu to iloczyn skalarny wektora wejsciowego z wektorem wagowym -
przy obu wektorach znormalizowanych jest to cosinus kata miedzy nimi, czyli
miara podobienstwa z przedzialu [0, 1].
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "siec"))
from NeuronNetwork import Network
from functions import linear, linear_derivative

DESCRIPTION_FILE = "description.txt"
BINARY_THRESHOLD = 128


class MadalineError(Exception):
    """Blad danych wejsciowych - komunikat idzie wprost do uzytkownika."""


# ------------------------------------------------------------------ obrazy

def image_to_vector(path, threshold=BINARY_THRESHOLD):
    """Obraz -> skala szarosci -> binaryzacja -> wektor o dlugosci 1.

    Tusz daje 1, tlo 0. Odwrotna konwencja zdominowalaby iloczyn skalarny tlem,
    ktore zajmuje wiekszosc obrazu, i wszystkie znaki wyszlyby identyczne.
    """
    if not os.path.isfile(path):
        raise MadalineError(f"Nie ma takiego pliku z obrazem: {path}")
    try:
        with Image.open(path) as handle:
            grayscale = handle.convert("L")
            pixels = np.asarray(grayscale, dtype=np.int16)
    except OSError as error:
        raise MadalineError(f"Nie udalo sie odczytac obrazu {path}: {error}")

    binary = (pixels < threshold).astype(float)   # tusz = 1, tlo = 0
    vector = binary.reshape(-1)
    length = np.linalg.norm(vector)
    if length == 0:
        raise MadalineError(
            f"Obraz {path} nie zawiera ani jednego ciemnego piksela, "
            f"wiec nie da sie go znormalizowac.")
    return vector / length, pixels.shape


# ------------------------------------------------- plik opisu description.txt

def read_description(directory):
    """Czyta description.txt i zwraca liste par (sciezka do pliku, etykieta).

    Format linii: <nazwa pliku>:<etykieta>
    """
    path = os.path.join(directory, DESCRIPTION_FILE)
    if not os.path.isfile(path):
        raise MadalineError(
            f"W katalogu {directory} brakuje pliku {DESCRIPTION_FILE}. "
            f"Wygeneruj znaki programem font_generator.py.")
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    entries = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if ":" not in line:
            raise MadalineError(
                f"{path}, linia {number}: brak separatora ':' w '{line}'.")
        file_name, label = line.split(":", 1)
        entries.append((os.path.join(directory, file_name.strip()), label.strip()))
    if not entries:
        raise MadalineError(f"Plik {path} nie zawiera zadnego wpisu.")
    return entries


def append_description(directory, file_name, label):
    """Dopisuje pojedyncza linie opisu, tworzac plik jesli jeszcze nie istnieje."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, DESCRIPTION_FILE), "a", encoding="utf-8") as handle:
        handle.write(f"{file_name}:{label}\n")


# ------------------------------------------------------------------ siec

class MadalineNetwork:
    """Warstwa MADALINE: Network([N, k]) bez biasu z aktywacja liniowa.

    Przy znormalizowanych wektorach wyjscie neuronu to cosinus kata miedzy jego
    wektorem wagowym a wejsciem, czyli miara podobienstwa z przedzialu [0, 1].
    """

    def __init__(self, network, labels, shape):
        self.network = network
        self.labels = labels
        self.shape = shape

    @classmethod
    def from_directory(cls, directory):
        """Buduje siec z description.txt - wagi to wprost znormalizowane wzorce,
        bez iteracyjnego treningu."""
        vectors, labels, shape = [], [], None
        for path, label in read_description(directory):
            vector, image_shape = image_to_vector(path)
            if shape is None:
                shape = image_shape
            elif image_shape != shape:
                raise MadalineError(
                    f"Obraz {path} ma rozdzielczosc {image_shape[1]}x{image_shape[0]}, "
                    f"a poprzednie maja {shape[1]}x{shape[0]}. Wszystkie znaki "
                    f"wzorcowe musza byc tej samej wielkosci.")
            vectors.append(vector)
            labels.append(label)

        network = Network([len(vectors[0]), len(vectors)], bias=False,
                          activations=[(linear, linear_derivative)])
        for neuron, vector in zip(network.layer_list[0].neurons, vectors):
            neuron.weights = vector
        return cls(network, labels, shape)

    @property
    def neuron_num(self):
        return len(self.labels)

    def outputs(self, input_vector):
        """Wyjscia wszystkich neuronow dla podanego wektora wejsciowego."""
        if input_vector.shape[0] != self.network.input_num:
            raise MadalineError(
                f"Wektor wejsciowy ma {input_vector.shape[0]} skladowych, "
                f"a neurony sieci {self.network.input_num}. Znaki testowe musza "
                f"miec te sama rozdzielczosc co wzorcowe "
                f"({self.shape[1]}x{self.shape[0]}).")
        self.network.forward(input_vector)
        return self.network.activations[-1]

    def recognize(self, input_vector):
        """Zwraca (etykieta neuronu zwycieskiego, wspolczynnik pewnosci)."""
        values = self.outputs(input_vector)
        winner = int(np.argmax(values))
        return self.labels[winner], float(values[winner])
