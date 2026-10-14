import numpy as np
import sys
from NeuronNetwork import Network
from file_service import save, load
from paths import PATH_TEST_DATA, PATH_TRAIN_DATA, PATH_SAVED, PATH_ASOCJACJA

# GLOBAL
neuron_network = None
network_structure = [4, 10, 10, 3]
use_bias = True


def ask_text(prompt):
    """Czyta linie z wejscia. Ctrl-D (EOF) konczy program zamiast rzucac EOFError."""
    try:
        return input(prompt)
    except EOFError:
        print("\nKoniec wejscia - zamykam program.")
        sys.exit()


def ask_int(prompt, minimum=None):
    """Czyta liczbe calkowita, powtarzajac pytanie zamiast wywalac program."""
    while True:
        try:
            value = int(ask_text(prompt))
        except ValueError:
            print("Podaj liczbe calkowita.")
            continue
        if minimum is not None and value < minimum:
            print(f"Wartosc musi byc nie mniejsza niz {minimum}.")
            continue
        return value


def ask_float(prompt):
    """Czyta liczbe zmiennoprzecinkowa, powtarzajac pytanie zamiast wywalac program."""
    while True:
        try:
            return float(ask_text(prompt))
        except ValueError:
            print("Podaj liczbe.")


def prepare_network(existing_network, structure, bias):
    """Zwraca siec do trenowania: istniejaca (dotrenowanie) albo swiezo utworzona.

    Bez tego kazde wejscie w trening tworzylo nowa siec, wiec wczytanie sieci
    z pliku i wybranie treningu po cichu wyrzucalo wczytany model.
    """
    if (existing_network is not None
            and existing_network.neurons_in_particular_layers == structure
            and existing_network.bias == bias):
        answer = ask_text("Istnieje juz siec o tej strukturze. Kontynuowac jej uczenie? [t/n]: ")
        if answer.strip().lower() in ("t", "tak"):
            print("Kontynuacja uczenia istniejacej sieci.")
            return existing_network
        print("Tworzenie nowej sieci.")
    return Network(structure, bias)


def choose_menu():
    while True:
        function_type = ask_int(f"""
        Domyslna siec neuronowa o parametrach {network_structure}
        Wybierz:
        [1] -> Stworz siec neuronowa
        [2] -> zapisz sieć neuronową do pliku
        [3] -> wczytaj sieć neuronową z pliku
        [4] -> trenuj sieć irysów
        [5] -> testuj sieć irystów
        [6] -> trenuj autoasocjację
        [7] -> testuj asocjację
        [8] -> zakoncz program: """)
        if 1 <= function_type <= 8:
            return function_type
        print("Wybierz opcje od 1 do 8.")


while True:
    output_menu = choose_menu()
    if output_menu == 1:
        input_neuron = ask_int("Podaj liczbe neuronych wejsciowych: ", minimum=1)
        network_structure = [input_neuron]
        amount_of_layers = ask_int("Podaj liczbe warstw ukrytych: ", minimum=0)
        for i in range(amount_of_layers):
            layer = ask_int(f"Podaj liczbe neuronow w {i + 1} warstwie: ", minimum=1)
            network_structure.append(layer)
        output_layer = ask_int("Podaj liczbe neuronow w warstwie wyjsciowej: ", minimum=1)
        network_structure.append(output_layer)
        use_bias = ask_text("Czy neurony maja miec bias? [t/n]: ").strip().lower() in ("t", "tak")
        neuron_network = Network(network_structure, use_bias)
    if output_menu == 2:
        if neuron_network is None:
            print("Brak sieci do zapisania - najpierw stworz ja [1] albo wytrenuj [4]/[6].")
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
        print(f"Wczytano siec {network_structure}, bias={use_bias}")
    if output_menu == 4:
        inputLearningRate = ask_float("Podaj wartosc learningRate: ")
        inputMomentum = ask_float("Podaj wartosc momentum: ")
        if (not (1 > inputMomentum >= 0)) or (not (1 > inputLearningRate > 0)):
            print("Podano błędne wartości")
            continue
        data_for_training = np.genfromtxt(PATH_TRAIN_DATA, delimiter=',')
        neuron_network = prepare_network(neuron_network, network_structure, use_bias)
        data_for_training = neuron_network.fit_normalization(data_for_training)
        neuron_network.train(data_for_training, epochs=150, shuffle=True, learning_rate=inputLearningRate, jump=2,
                             given_error=0,
                             momentum=inputMomentum, is_last_column_class=True)

    if output_menu == 5:
        if neuron_network is None:
            print("Brak sieci do przetestowania - najpierw wytrenuj [4] albo wczytaj [3].")
            continue
        data_for_testing = np.genfromtxt(PATH_TEST_DATA, delimiter=',')
        data_for_testing = neuron_network.normalize(data_for_testing)
        neuron_network.test(data_for_testing, True)
    if output_menu == 6:
        network_structure = [4, 2, 4]
        inputLearningRate = ask_float("Podaj wartosc learningRate: ")
        inputMomentum = ask_float("Podaj wartosc momentum: ")
        if (not (1 > inputMomentum >= 0)) or (not (1 > inputLearningRate > 0)):
            print("Podano błędne wartości")
            continue
        data_for_training = np.genfromtxt(PATH_ASOCJACJA, delimiter=',')
        neuron_network = prepare_network(neuron_network, network_structure, use_bias)
        neuron_network.train(data_for_training, epochs=1000, shuffle=True, learning_rate=inputLearningRate, jump=1,
                             given_error=0,
                             momentum=inputMomentum, is_last_column_class=False)
    if output_menu == 7:
        if neuron_network is None:
            print("Brak sieci do przetestowania - najpierw wytrenuj [6] albo wczytaj [3].")
            continue
        data_for_testing = np.genfromtxt(PATH_ASOCJACJA, delimiter=',')
        neuron_network.test(data_for_testing, False)
    if output_menu == 8:
        sys.exit()
