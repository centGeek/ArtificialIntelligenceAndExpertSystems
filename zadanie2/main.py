import numpy as np
import sys
from NeuronNetwork import Network
from file_service import save, load
from paths import PATH_TEST_DATA, PATH_TRAIN_DATA, PATH_SAVED, PATH_ASOCJACJA

# GLOBAL
neuron_network = None
network_structure = [4, 10, 10, 3]


def choose_menu():
    while True:
        function_type = input(f"""
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
        return function_type


while True:
    output_menu = int(choose_menu())
    if output_menu == 1:
        input_neuron = int(input("Podaj liczbe neuronych wejsciowych: "))
        network_structure = [input_neuron]
        amount_of_layers = int(input("Podaj liczbe warstw ukrytych: "))
        for i in range(amount_of_layers):
            layer = int(input(f"Podaj liczbe neuronow w {i + 1} warstwie: "))
            network_structure.append(layer)
        output_layer = int(input("Podaj liczbe neuronow w warstwie wyjsciowej: "))
        network_structure.append(output_layer)
        neuron_network = Network(network_structure, False)
    if output_menu == 2:
        save(neuron_network, PATH_SAVED)
    if output_menu == 3:
        neuron_network = load(PATH_SAVED)
        network_structure = neuron_network.neurons_in_particular_layers
    if output_menu == 4:
        inputLearningRate = float(input("Podaj wartosc learningRate: "))
        inputMomentum = float(input("Podaj wartosc momentum: "))
        if (not (1 > inputMomentum >= 0)) or (not (1 > inputLearningRate > 0)):
            print("Podano błędne wartości")
            continue
        data_for_training = np.genfromtxt(PATH_TRAIN_DATA, delimiter=',')
        neuron_network = Network(network_structure, True)
        neuron_network.train(data_for_training, epochs=150, shuffle=True, learning_rate=inputLearningRate, jump=2,
                             given_error=0,
                             momentum=inputMomentum, is_last_column_class=True)

    if output_menu == 5:
        data_for_testing = np.genfromtxt(PATH_TEST_DATA, delimiter=',')
        neuron_network.test(data_for_testing, True)
    if output_menu == 6:
        network_structure = [4, 2, 4]
        inputLearningRate = float(input("Podaj wartosc learningRate: "))
        inputMomentum = float(input("Podaj wartosc momentum: "))
        if (not (1 > inputMomentum >= 0)) or (not (1 > inputLearningRate > 0)):
            print("Podano błędne wartości")
            continue
        data_for_training = np.genfromtxt(PATH_ASOCJACJA, delimiter=',')
        neuron_network = Network(network_structure, True)
        neuron_network.train(data_for_training, epochs=1000, shuffle=True, learning_rate=inputLearningRate, jump=1,
                             given_error=0,
                             momentum=inputMomentum, is_last_column_class=False)
    if output_menu == 7:
        data_for_testing = np.genfromtxt(PATH_ASOCJACJA, delimiter=',')
        neuron_network.test(data_for_testing, False)
    if output_menu == 8:
        sys.exit()
