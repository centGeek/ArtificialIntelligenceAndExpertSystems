import pandas as pd
from file_service import save, load
from NeuronNetwork import NeuronNetwork
import sys
layers = []
choice = input("Chcesz stworzyc siec? Domyslne wartosci to 4, 7, 3. y/n\n")
if(choice == 'y'):
    layers_amount = input("Podaj liczbe warstw\n")
    for i in range(0, int(layers_amount)):
        layers.append(int(input(f"Podaj {i+1} warstwe\n")))
    neurons = NeuronNetwork(layers, True)
else: 
    neurons = NeuronNetwork([4,7,3], True)
while(True):
    choice = input("Wybierz: \n1 -> tryb treningu\n2 -> tryb testowania\n3 -> zapisz do pliku\n4 -> odczytaj z pliku\n5 -> koniec programu\n")   
    if(choice == '3'):
        save(neurons, "./zadanie2/files/saved_network.txt")
    if(choice == '4'):
        load("./zadanie2/files/saved_network.txt")
    if(choice == '5'):
        sys.exit()