import os
import sys

import numpy as np

from NeuronNetwork import Network
from dataset import (DatasetError, FORMAT_CLASS, FORMAT_IO,
                     from_class_file, from_io_file, split)
from file_service import save, load
from paths import PATH_ASOCJACJA, PATH_SAVED, PATH_TEST_DATA, PATH_TRAIN_DATA

# GLOBAL
EFFECTIVE_STEP_LIMIT = 10.0
neuron_network = None
network_structure = [4, 10, 10, 3]
use_bias = True
train_set = None
test_set = None
normalize_features = True

HELP = """
Program uczy siec MLP metoda wstecznej propagacji bledu z momentum.
Dziala na dowolnym zbiorze danych w formacie CSV.

Kolejnosc pracy: [6] wybierz zbior -> [1] stworz siec -> [4] trenuj -> [5] testuj.
W kazdym pytaniu mozesz wpisac '?', zeby zobaczyc te pomoc.

Obslugiwane formaty plikow CSV (naglowek, jesli jest, zostanie pominiety):
  [1] klasyfikacja - kolumny cech, ostatnia kolumna to numer klasy 0, 1, 2, ...
      przyklad: 5.1,3.5,1.4,0.2,0
  [2] wejscia i wyjscia - N pierwszych kolumn to wejscia, reszta to wzorzec
      przyklad: 1,0,0,0,1,0,0,0  (przy 4 wejsciach)

Liczba neuronow wejsciowych musi rownac sie liczbie kolumn wejsciowych zbioru,
a liczba wyjsciowych liczbie klas albo kolumn wzorca.

Normalizacja cech (z-score) liczona jest tylko na zbiorze treningowym i potem
stosowana do testowego. Warto ja wlaczyc, gdy cechy maja rozne rzedy wielkosci
(irysy), nie ma sensu dla danych zero-jedynkowych (autoasocjacja).
"""


def ask_text(prompt):
    """Czyta linie z wejscia. '?' pokazuje pomoc, Ctrl-D konczy program."""
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            print("\nKoniec wejscia - zamykam program.")
            sys.exit()
        if value == "?":
            print(HELP)
            continue
        return value


def ask_int(prompt, minimum=None, default=None):
    """Czyta liczbe calkowita, powtarzajac pytanie zamiast wywalac program."""
    while True:
        raw = ask_text(prompt)
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Podaj liczbe calkowita.")
            continue
        if minimum is not None and value < minimum:
            print(f"Wartosc musi byc nie mniejsza niz {minimum}.")
            continue
        return value


def ask_float(prompt, default=None):
    """Czyta liczbe zmiennoprzecinkowa, powtarzajac pytanie zamiast wywalac program."""
    while True:
        raw = ask_text(prompt)
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Podaj liczbe.")


def confirm_effective_step(learning_rate, momentum):
    """Ostrzega, gdy lr/(1-momentum) wchodzi w zakres, w ktorym uczenie sie rozpada.

    Przy momentum kolejne poprawki sumuja sie geometrycznie, wiec realna dlugosc
    kroku to lr/(1-momentum), a nie samo learning_rate. Pomiary na tym projekcie
    (8 seedow, 150 epok, irysy po normalizacji): krok 9 -> 98.75%, krok 11.25 ->
    92.50%, krok 12.86 -> 72.08%. Stad prog 10.
    """
    effective = learning_rate / (1 - momentum)
    if effective <= EFFECTIVE_STEP_LIMIT:
        return True
    print(f"\nUwaga: krok efektywny lr/(1-momentum) = {effective:.2f}, "
          f"czyli powyzej bezpiecznego {EFFECTIVE_STEP_LIMIT:.0f}.")
    print(f"Zejscie do granicy wymaga przy momentum={momentum} "
          f"wartosci learning_rate={EFFECTIVE_STEP_LIMIT * (1 - momentum):.3f} lub mniejszej.")
    return ask_text("Kontynuowac mimo to? [t/n]: ").lower() in ("t", "tak")


def prepare_network(existing_network, structure, bias):
    """Zwraca siec do trenowania: istniejaca (dotrenowanie) albo swiezo utworzona.

    Bez tego kazde wejscie w trening tworzylo nowa siec, wiec wczytanie sieci
    z pliku i wybranie treningu po cichu wyrzucalo wczytany model.
    """
    if (existing_network is not None
            and existing_network.neurons_in_particular_layers == structure
            and existing_network.bias == bias):
        if ask_text("Istnieje juz siec o tej strukturze. Kontynuowac jej uczenie? [t/n]: "
                    ).lower() in ("t", "tak"):
            print("Kontynuacja uczenia istniejacej sieci.")
            return existing_network
        print("Tworzenie nowej sieci.")
    return Network(structure, bias)


def read_dataset(prompt, data_format=None, input_num=None, output_num=None):
    """Pyta o plik i wczytuje zbior. Zwraca None, jesli sie nie udalo."""
    path = os.path.expanduser(ask_text(prompt))
    if data_format is None:
        choice = ask_text("Format: [1] klasyfikacja / [2] wejscia+wyjscia: ")
        data_format = FORMAT_CLASS if choice == "1" else FORMAT_IO
    try:
        if data_format == FORMAT_CLASS:
            dataset, header = from_class_file(path, output_num)
        else:
            if input_num is None:
                input_num = ask_int("Ile pierwszych kolumn to wejscia: ", minimum=1)
            dataset, header = from_io_file(path, input_num)
    except (DatasetError, OSError, ValueError) as error:
        print(f"Blad: {error}")
        return None
    print(f"Wczytano {dataset.summary()}" + (" (pominieto naglowek)" if header else ""))
    return dataset


def matches_dataset(structure, dataset):
    """Sprawdza, czy struktura sieci zgadza sie wymiarami ze zbiorem."""
    if dataset is None:
        return True
    if structure[0] != dataset.input_num or structure[-1] != dataset.output_num:
        print(f"Siec {structure} nie pasuje do zbioru ({dataset.input_num} wejsc, "
              f"{dataset.output_num} wyjsc). Zmien strukture opcja [1].")
        return False
    return True


def choose_menu():
    while True:
        print(f"""
        Siec: {network_structure}, bias={'tak' if use_bias else 'nie'}
        Treningowy: {train_set.summary() if train_set else 'nie wybrano'}
        Testowy:    {test_set.summary() if test_set else 'nie wybrano'}
        Normalizacja cech: {'tak' if normalize_features else 'nie'}

        [1] -> Stworz siec neuronowa
        [2] -> Zapisz siec do pliku
        [3] -> Wczytaj siec z pliku
        [4] -> Trenuj siec
        [5] -> Testuj siec
        [6] -> Wybierz zbior danych
        [7] -> Przelacz normalizacje cech
        [8] -> Zakoncz program""")
        value = ask_text("Wybierz ('?' = pomoc): ")
        if value in ("1", "2", "3", "4", "5", "6", "7", "8"):
            return int(value)
        print("Wybierz opcje od 1 do 8.")


while True:
    output_menu = choose_menu()
    if output_menu == 1:
        if train_set is not None:
            print(f"Zbior treningowy wymaga {train_set.input_num} wejsc "
                  f"i {train_set.output_num} wyjsc (Enter przyjmuje te wartosci).")
        default_in = train_set.input_num if train_set else None
        default_out = train_set.output_num if train_set else None
        input_neuron = ask_int("Podaj liczbe neuronow wejsciowych: ", 1, default_in)
        network_structure = [input_neuron]
        amount_of_layers = ask_int("Podaj liczbe warstw ukrytych: ", 0)
        for i in range(amount_of_layers):
            network_structure.append(ask_int(f"Podaj liczbe neuronow w {i + 1} warstwie: ", 1))
        network_structure.append(ask_int("Podaj liczbe neuronow wyjsciowych: ", 1, default_out))
        use_bias = ask_text("Czy neurony maja miec bias? [t/n]: ").lower() in ("t", "tak")
        neuron_network = Network(network_structure, use_bias)
        matches_dataset(network_structure, train_set)
    if output_menu == 2:
        if neuron_network is None:
            print("Brak sieci do zapisania - najpierw stworz ja [1] albo wytrenuj [4].")
            continue
        save(neuron_network, PATH_SAVED)
        print(f"Zapisano siec do {PATH_SAVED}")
    if output_menu == 3:
        try:
            loaded_network = load(PATH_SAVED)
        except (OSError, EOFError) as error:
            print(f"Nie udalo sie wczytac sieci: {error}")
            continue
        if not isinstance(loaded_network, Network):
            print(f"Plik {PATH_SAVED} nie zawiera sieci neuronowej.")
            continue
        neuron_network = loaded_network
        network_structure = neuron_network.neurons_in_particular_layers
        use_bias = neuron_network.bias
        normalize_features = getattr(neuron_network, "feature_mean", None) is not None
        print(f"Wczytano siec {network_structure}, bias={use_bias}")
    if output_menu == 4:
        if train_set is None:
            print("Brak zbioru treningowego - wybierz go opcja [6].")
            continue
        if not matches_dataset(network_structure, train_set):
            continue
        inputLearningRate = ask_float("Podaj wartosc learningRate: ")
        inputMomentum = ask_float("Podaj wartosc momentum: ")
        if (not (1 > inputMomentum >= 0)) or (not (1 > inputLearningRate > 0)):
            print("Podano błędne wartości")
            continue
        inputEpochs = ask_int("Podaj liczbe epok [150]: ", 1, 150)
        inputError = ask_float("Prog bledu konczacy uczenie [0 = pelna liczba epok]: ", 0.0)
        if not confirm_effective_step(inputLearningRate, inputMomentum):
            continue
        neuron_network = prepare_network(neuron_network, network_structure, use_bias)
        data_for_training = train_set.data.copy()
        if normalize_features:
            data_for_training = neuron_network.fit_normalization(data_for_training)
        else:
            neuron_network.disable_normalization()
        try:
            neuron_network.train(data_for_training, epochs=inputEpochs, shuffle=True,
                                 learning_rate=inputLearningRate, jump=2,
                                 given_error=inputError, momentum=inputMomentum,
                                 is_last_column_class=train_set.format == FORMAT_CLASS)
        except ValueError as error:
            print(f"Blad podczas uczenia: {error}")
    if output_menu == 5:
        if neuron_network is None:
            print("Brak sieci do przetestowania - najpierw wytrenuj [4] albo wczytaj [3].")
            continue
        dataset = test_set or train_set
        if dataset is None:
            print("Brak zbioru testowego - wybierz go opcja [6].")
            continue
        try:
            neuron_network.test(neuron_network.normalize(dataset.data),
                                dataset.format == FORMAT_CLASS)
        except ValueError as error:
            print(f"Blad podczas testowania: {error}")
    if output_menu == 6:
        print("""
        [1] -> Irysy (wbudowane, 4 cechy -> 3 klasy)
        [2] -> Autoasocjacja (wbudowana, 4 wejscia -> 4 wyjscia)
        [3] -> Wlasne pliki: osobno treningowy i testowy
        [4] -> Wlasny plik podzielony na czesc treningowa i testowa""")
        source = ask_text("Wybierz zbior: ")
        try:
            if source == "1":
                train_set, _ = from_class_file(PATH_TRAIN_DATA)
                test_set, _ = from_class_file(PATH_TEST_DATA, train_set.output_num)
                normalize_features = True
            elif source == "2":
                train_set, _ = from_io_file(PATH_ASOCJACJA, 4)
                test_set, _ = from_io_file(PATH_ASOCJACJA, 4)
                normalize_features = False
            elif source == "3":
                dataset = read_dataset("Plik treningowy: ")
                if dataset is None:
                    continue
                train_set = dataset
                test_set = read_dataset("Plik testowy: ", train_set.format,
                                        train_set.input_num, train_set.output_num)
                normalize_features = train_set.format == FORMAT_CLASS
            elif source == "4":
                dataset = read_dataset("Plik do podzialu: ")
                if dataset is None:
                    continue
                ratio = ask_float("Jaka czesc na test [0.2]: ", 0.2)
                train_set, test_set = split(dataset, ratio)
                normalize_features = dataset.format == FORMAT_CLASS
                print(f"Podzielono: {len(train_set.data)} treningowych, "
                      f"{len(test_set.data)} testowych.")
            else:
                continue
        except DatasetError as error:
            print(f"Blad: {error}")
            continue
        print(f"Normalizacja cech ustawiona na "
              f"{'tak' if normalize_features else 'nie'} (zmiana opcja [7]).")
        matches_dataset(network_structure, train_set)
    if output_menu == 7:
        normalize_features = not normalize_features
        print(f"Normalizacja cech: {'wlaczona' if normalize_features else 'wylaczona'}.")
    if output_menu == 8:
        sys.exit()
