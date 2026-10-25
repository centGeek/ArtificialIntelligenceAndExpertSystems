"""Wczytywanie zbiorow danych wskazanych przez uzytkownika."""

import os

import numpy as np

FORMAT_CLASS = "class"
FORMAT_IO = "io"


class DatasetError(Exception):
    """Blad w pliku ze zbiorem - komunikat idzie wprost do uzytkownika."""


def load_matrix(path, delimiter=","):
    """Wczytuje CSV do macierzy liczb, pomijajac ewentualny naglowek.

    np.genfromtxt zamienia tekstowy naglowek na wiersz NaN, ktory bez tej
    kontroli wszedlby do uczenia jako probka z samych NaN-ow.
    """
    if not os.path.isfile(path):
        raise DatasetError(f"Nie ma takiego pliku: {path}")
    data = np.atleast_2d(np.genfromtxt(path, delimiter=delimiter))
    if data.size == 0:
        raise DatasetError(f"Plik {path} jest pusty.")
    header = bool(np.isnan(data[0]).all())
    if header:
        data = np.atleast_2d(np.genfromtxt(path, delimiter=delimiter, skip_header=1))
    if data.size == 0 or np.isnan(data).any():
        raise DatasetError(f"Plik {path} zawiera puste lub nieliczbowe pola.")
    if data.shape[1] < 2:
        raise DatasetError(f"Plik {path} ma tylko jedna kolumne.")
    return data, header


class Dataset:
    """Macierz probek razem z informacja, jak ja czytac."""

    def __init__(self, path, data_format, data, input_num, output_num):
        self.path = path
        self.format = data_format
        self.data = data
        self.input_num = input_num
        self.output_num = output_num

    def summary(self):
        kind = "cech -> klas" if self.format == FORMAT_CLASS else "wejsc -> wyjsc"
        left, right = kind.split(" -> ")
        return (f"{os.path.basename(self.path)} [{self.input_num} {left} -> "
                f"{self.output_num} {right}, {len(self.data)} probek]")


def from_class_file(path, output_num=None):
    """Zbior klasyfikacyjny: kolumny cech + ostatnia kolumna z numerem klasy.

    output_num pozwala wymusic liczbe klas. Jest potrzebne dla zbioru testowego,
    w ktorym moze brakowac najwyzszej klasy.
    """
    data, header = load_matrix(path)
    labels = data[:, -1]
    if np.any(labels != np.floor(labels)) or np.any(labels < 0):
        raise DatasetError(
            f"Plik {path}: ostatnia kolumna musi zawierac calkowite numery klas 0, 1, 2, ...")
    detected = int(labels.max()) + 1
    if output_num is None:
        output_num = detected
    elif detected > output_num:
        raise DatasetError(
            f"Plik {path} zawiera klase {detected - 1}, a oczekiwano najwyzej {output_num - 1}.")
    return Dataset(path, FORMAT_CLASS, data, data.shape[1] - 1, output_num), header


def from_io_file(path, input_num):
    """Zbior z jawnym wzorcem: input_num kolumn wejsc, reszta kolumn to wyjscia."""
    data, header = load_matrix(path)
    if not 0 < input_num < data.shape[1]:
        raise DatasetError(
            f"Plik {path} ma {data.shape[1]} kolumn, wiec liczba wejsc musi byc "
            f"z zakresu 1..{data.shape[1] - 1}, a podano {input_num}.")
    return Dataset(path, FORMAT_IO, data, input_num, data.shape[1] - input_num), header


def split(dataset, test_ratio, seed=None):
    """Dzieli zbior na czesc treningowa i testowa, losujac kolejnosc probek.

    Ksztalt wyjscia jest przepisywany, a nie liczony od nowa, zeby czesc
    testowa bez jakiejs klasy nie dostala wezszego kodowania niz treningowa.
    """
    total = len(dataset.data)
    test_size = int(round(total * test_ratio))
    if test_size == 0 or test_size == total:
        raise DatasetError(f"Przy {total} probkach proporcja {test_ratio} daje pusta czesc.")
    order = np.random.default_rng(seed).permutation(total)
    def part(rows, suffix):
        return Dataset(f"{dataset.path} ({suffix})", dataset.format, rows,
                       dataset.input_num, dataset.output_num)
    return (part(dataset.data[order[test_size:]], "trening"),
            part(dataset.data[order[:test_size]], "test"))
