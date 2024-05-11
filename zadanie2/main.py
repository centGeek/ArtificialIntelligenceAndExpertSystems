import pandas as pd
from file_service import save, load
choice = input("Wybierz: \n1 -> tryb treningu\n2 -> tryb testowania\n3 -> zapisz do pliku\n 4 -> odczytaj z pliku")
if(choice == '3'):
    save("./zadanie2/files/saved_network.txt")
if(choice == '4'):
    load("./zadanie2/files/saved_network.txt")