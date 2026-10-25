"""Wczytywanie i walidacja zbiorow danych podawanych przez uzytkownika.

Wczesniej zbiory byly zaszyte w main.py (irysy + autoasocjacja), a liczba cech
i klas byla zalozona z gory. Tutaj kazdy zbior sam opisuje swoj ksztalt, dzieki
czemu CLI moze sprawdzic, czy pasuje do struktury sieci, zanim zacznie uczenie.
"""

import os

import numpy as np

FORMAT_CLASS = "class"
FORMAT_IO = "io"

FORMAT_NAMES = {
    FORMAT_CLASS: "klasyfikacja - ostatnia kolumna to numer klasy (0, 1, 2, ...)",
    FORMAT_IO: "wejscia + wyjscia - N pierwszych kolumn to wejscia, reszta to wzorzec",
}


class DatasetError(Exception):
    """Blad w pliku ze zbiorem - komunikat jest pisany wprost do uzytkownika."""


def load_matrix(path, delimiter=","):
    """Wczytuje CSV do macierzy liczb, sam pomijajac ewentualny naglowek.

    np.genfromtxt zamienia tekstowy naglowek na wiersz NaN. Bez tej kontroli
    naglowek wchodzilby do uczenia jako probka zlozona z samych NaN-ow i siec
    po cichu psulaby sobie wszystkie wagi.
    """
    if not os.path.isfile(path):
        raise DatasetError(f"Nie ma takiego pliku: {path}")
    try:
        data = np.genfromtxt(path, delimiter=delimiter)
    except (OSError, ValueError) as error:
        raise DatasetError(f"Nie udalo sie wczytac {path}: {error}")
    data = np.atleast_2d(data)
    if data.size == 0:
        raise DatasetError(f"Plik {path} jest pusty.")

    header_skipped = False
    if np.isnan(data[0]).all():
        data = np.atleast_2d(np.genfromtxt(path, delimiter=delimiter, skip_header=1))
        header_skipped = True
        if data.size == 0:
            raise DatasetError(f"Plik {path} zawiera tylko naglowek, bez danych.")

    if np.isnan(data).any():
        first_bad = int(np.argwhere(np.isnan(data).any(axis=1))[0][0])
        line = first_bad + 1 + int(header_skipped)
        raise DatasetError(
            f"Plik {path}: wiersz {line} ma puste lub nieliczbowe pole. "
            f"Oczekiwany jest CSV z samymi liczbami.")
    if data.shape[1] < 2:
        raise DatasetError(f"Plik {path} ma tylko jedna kolumne - za malo na zbior uczacy.")
    return data, header_skipped


class Dataset:
    """Macierz probek razem z informacja, jak ja czytac."""

    def __init__(self, path, data_format, data, input_num, output_num,
                 header_skipped=False, missing_classes=()):
        self.path = path
        self.format = data_format
        self.data = data
        self.input_num = input_num
        self.output_num = output_num
        self.header_skipped = header_skipped
        self.missing_classes = list(missing_classes)

    @property
    def sample_num(self):
        return len(self.data)

    @property
    def name(self):
        return os.path.basename(self.path)

    def shape_text(self):
        if self.format == FORMAT_CLASS:
            return f"{self.input_num} cech -> {self.output_num} klas"
        return f"{self.input_num} wejsc -> {self.output_num} wyjsc"

    def summary(self):
        return f"{self.name} [{self.shape_text()}, {self.sample_num} probek]"

    def suggested_structure(self, hidden=(10, 10)):
        """Proponowana struktura sieci dopasowana wymiarami do tego zbioru."""
        return [self.input_num, *hidden, self.output_num]

    def class_counts(self):
        """Liczba probek w kazdej klasie - tylko dla formatu klasyfikacyjnego."""
        if self.format != FORMAT_CLASS:
            return None
        labels = self.data[:, -1].astype(int)
        return [int(np.count_nonzero(labels == c)) for c in range(self.output_num)]


def from_class_file(path, delimiter=",", output_num=None):
    """Zbior klasyfikacyjny: kolumny cech + ostatnia kolumna z numerem klasy.

    output_num pozwala wymusic szerokosc kodowania 1-z-N. Jest potrzebne dla
    zbioru testowego, w ktorym moze brakowac najwyzszej klasy - bez tego test
    mialby wezsze wyjscie niz siec i wzorce by sie nie zgadzaly.
    """
    data, header_skipped = load_matrix(path, delimiter)
    labels = data[:, -1]
    if np.any(labels != np.floor(labels)) or np.any(labels < 0):
        raise DatasetError(
            f"Plik {path}: ostatnia kolumna musi zawierac calkowite numery klas "
            f"0, 1, 2, ..., a zawiera np. {labels[np.argmax(labels != np.floor(labels))]}.")
    detected = int(labels.max()) + 1
    if output_num is None:
        output_num = detected
    elif detected > output_num:
        raise DatasetError(
            f"Plik {path} zawiera klase {detected - 1}, a oczekiwano najwyzej {output_num - 1}.")
    missing = [c for c in range(output_num) if not np.any(labels == c)]
    return Dataset(path, FORMAT_CLASS, data, data.shape[1] - 1, output_num,
                   header_skipped, missing)


def from_io_file(path, input_num, delimiter=","):
    """Zbior z jawnym wzorcem: input_num kolumn wejsc, reszta kolumn to wyjscia."""
    data, header_skipped = load_matrix(path, delimiter)
    columns = data.shape[1]
    if not 0 < input_num < columns:
        raise DatasetError(
            f"Plik {path} ma {columns} kolumn, wiec liczba wejsc musi byc z zakresu "
            f"1..{columns - 1}, a podano {input_num}.")
    return Dataset(path, FORMAT_IO, data, input_num, columns - input_num, header_skipped)


def split(dataset, test_ratio, seed=None):
    """Dzieli zbior na czesc treningowa i testowa, losujac kolejnosc probek.

    Zwraca dwa obiekty Dataset o tym samym ksztalcie co zrodlowy - szerokosc
    wyjscia jest przepisywana, a nie liczona od nowa, zeby czesc testowa bez
    jakiejs klasy nie dostala wezszego kodowania niz treningowa.
    """
    if not 0 < test_ratio < 1:
        raise DatasetError("Proporcja czesci testowej musi byc z przedzialu (0, 1).")
    total = dataset.sample_num
    test_size = int(round(total * test_ratio))
    if test_size == 0 or test_size == total:
        raise DatasetError(
            f"Przy {total} probkach proporcja {test_ratio} daje pusta czesc. "
            f"Wybierz inna proporcje.")

    order = np.random.default_rng(seed).permutation(total)
    test_rows = dataset.data[order[:test_size]]
    train_rows = dataset.data[order[test_size:]]

    def part(rows, suffix):
        missing = ()
        if dataset.format == FORMAT_CLASS:
            labels = rows[:, -1]
            missing = [c for c in range(dataset.output_num) if not np.any(labels == c)]
        return Dataset(f"{dataset.path} ({suffix})", dataset.format, rows,
                       dataset.input_num, dataset.output_num,
                       dataset.header_skipped, missing)

    return part(train_rows, "czesc treningowa"), part(test_rows, "czesc testowa")
