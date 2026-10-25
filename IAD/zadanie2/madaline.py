"""Siec MADALINE do rozpoznawania znakow oraz obsluga obrazow i plikow opisu.

Siec sklada sie z jednej warstwy neuronow liniowych. Kazdy neuron odpowiada za
jeden znak wzorcowy, a jego wektor wagowy to znormalizowany wektor tego znaku.
Wyjscie neuronu to iloczyn skalarny wektora wejsciowego z wektorem wagowym -
przy obu wektorach znormalizowanych jest to cosinus kata miedzy nimi, czyli
miara podobienstwa z przedzialu [0, 1].
"""

import os

import numpy as np
from PIL import Image

DESCRIPTION_FILE = "description.txt"
BINARY_THRESHOLD = 128


class MadalineError(Exception):
    """Blad danych wejsciowych - komunikat idzie wprost do uzytkownika."""


# ------------------------------------------------------------------ obrazy

def image_to_vector(path, threshold=BINARY_THRESHOLD):
    """Zamienia obraz znaku na znormalizowany wektor wejsciowy sieci.

    Kolejnosc operacji: skala szarosci -> binaryzacja progiem -> wektoryzacja
    (sklejenie wierszy jeden za drugim) -> normalizacja do dlugosci 1.

    Piksele ciemne (tusz litery) daja 1, jasne (tlo) daja 0. Zgadza sie to
    z materialem teoretycznym, gdzie ksztalt litery kodowany jest jedynkami.
    Odwrotna konwencja bylaby bledem merytorycznym: tlo zajmuje wiekszosc
    obrazu, wiec zdominowaloby iloczyn skalarny i wszystkie znaki wyszlyby
    niemal identyczne.
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
    """Jedna warstwa neuronow liniowych - po jednym na znak wzorcowy."""

    def __init__(self, weights, labels, shape):
        self.weights = weights          # macierz: wiersz = wektor wagowy neuronu
        self.labels = labels
        self.shape = shape

    @classmethod
    def from_directory(cls, directory):
        """Buduje siec ze znakow wzorcowych opisanych w description.txt.

        Wektor wagowy neuronu ustawiany jest wprost na znormalizowany wzorzec
        - siec nie wymaga iteracyjnego treningu, wagi wynikaja z danych.
        """
        entries = read_description(directory)
        vectors, labels, shape = [], [], None
        for path, label in entries:
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
        return cls(np.array(vectors), labels, shape)

    @property
    def neuron_num(self):
        return len(self.labels)

    def outputs(self, input_vector):
        """Wyjscia wszystkich neuronow dla podanego wektora wejsciowego."""
        if input_vector.shape[0] != self.weights.shape[1]:
            raise MadalineError(
                f"Wektor wejsciowy ma {input_vector.shape[0]} skladowych, "
                f"a neurony sieci {self.weights.shape[1]}. Znaki testowe musza "
                f"miec te sama rozdzielczosc co wzorcowe "
                f"({self.shape[1]}x{self.shape[0]}).")
        return self.weights @ input_vector

    def recognize(self, input_vector):
        """Zwraca (etykieta neuronu zwycieskiego, wspolczynnik pewnosci)."""
        values = self.outputs(input_vector)
        winner = int(np.argmax(values))
        return self.labels[winner], float(values[winner])
