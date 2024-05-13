import numpy as np
import pickle
import sys
from NeuronNetwork import Network
from file_service import save, load
from paths import PATH_TEST_DATA, PATH_TRAIN_DATA, PATH_SAVED
neuron_network = None
network_structure = [4, 10, 10,  3]
def choose_menu():
    while True:
        function_type = input(f"""
        Domyslna siec neuronowa o parametrach {network_structure}
        Wybierz:
        [1] -> Stworz siec neuronowa
        [2] -> zapisz sieć neuronową do pliku
        [3] -> wczytaj sieć neuronową z pliku
        [4] -> trenuj sieć 
        [5] -> testuj sieć
        [6] -> zakoncz program: """)
        return function_type


while True:
    output_menu = int(choose_menu())
    if output_menu == 1:
        input_neuron = int(input("Podaj liczbe neuronych wejsciowych: "))
        network_structure = [input_neuron]
        amount_of_layers = int(input("Podaj liczbe warstw ukrytych: "))
        for i in range(amount_of_layers):
            layer = int(input(f"Podaj liczbe neuronow w {i+1} warstwie: "))
            network_structure.append(layer)
        output_layer = int(input("Podaj liczbe neuronow w warstwie wyjsciowej: "))
        network_structure.append(output_layer)
        neuron_network = Network(network_structure, False)
    if output_menu == 2:
        filename = input("podaj nazwe pliku,do którego chcesz zapisać sieć, bez rozszerenia: ") + ".pkl"
        save(neuron_network, "")
    if output_menu == 3:
        load(PATH_SAVED)
    if output_menu == 4:
        which_data_set = int(input("[1] - irysy\n[2] - autoasocjacja\n[3] - własny zbiór: """))
        data_for_training = None
        if which_data_set == 1:
            data_for_training = np.genfromtxt(PATH_TRAIN_DATA, delimiter=',')
            np.random.shuffle(data_for_training)  # mieszam to bo jest ustawione gatunkami narazie
        elif which_data_set == 2:
            data_for_training = np.genfromtxt(PATH_TRAIN_DATA, delimiter=',')
        else:
            data_for_training = np.loadtxt(input("podaj nazwę pliku z rozszerzeniem txt: "))
        neuron_network = Network(network_structure, False)
        neuron_network.train(data_for_training, epochs=10, shuffle=True, learning_rate=1, jump=1, given_error=0.1, momentum=0)
        print("statystyki zapisano do pliku!")

    if output_menu == 5:
        which_data_set = int(input("""
            [1] - zbiór irysów
            [2] - własny zbiór: """))
        data_for_testing = None
        if which_data_set == 1:
            data_for_testing = np.genfromtxt(PATH_TEST_DATA, delimiter=',')
        elif which_data_set == 2:
            data_for_testing = np.genfromtxt('Data/autoasocjacja.csv', delimiter=',')
        neuron_network = Network(network_structure, False)
        neuron_network.test(data_for_testing)
    if output_menu == 6:
        sys.exit()