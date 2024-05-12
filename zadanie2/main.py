import pandas as pd
from file_service import save, load
from NeuronNetwork import NeuronNetwork
from network_algorithms import Network
from data_preprocessing import load_train_and_test
import sys
layers = []
x = 4
y = 10 
z = 3
choice = input(f"Chcesz stworzyc siec? Domyslne wartosci to {x}, {y}, {z}. y/n\n")
if(choice == 'y'):
    layers_amount = input("Podaj liczbe warstw\n")
    for i in range(0, int(layers_amount)):
        layers.append(int(input(f"Podaj {i+1} warstwe\n")))
    neurons_network = NeuronNetwork(layers, True)
else: 
    neurons_network = NeuronNetwork([x,y,z], True)
while(True):
    choice = input("Wybierz: \n1 -> tryb treningu\n2 -> tryb testowania\n3 -> zapisz do pliku\n4 -> odczytaj z pliku\n5 -> koniec programu\n")   
    if(choice == '1'):
        
        net_train = Network(neurons_network)
        train_data, test_data = load_train_and_test()
        net_train.SGD(train_data, epochs=15, mini_batch_size=10, learning_rate=0.1,desired_precision=1, momentum=0.9, test_data=test_data)
        correct_results2 = net_train.evaluate(test_data)
        total_samples2 = len(test_data)
        accuracy2 = correct_results2 / total_samples2
        print(f"Accuracy without momentum: {accuracy2 * 100}%")
    if(choice == '3'):
        save(neurons_network, "./zadanie2/files/saved_network.txt")
    if(choice == '4'):
        net = load("./zadanie2/files/saved_network.txt")
        print(net)
    if(choice == '5'):
        sys.exit()