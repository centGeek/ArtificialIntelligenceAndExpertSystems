import matplotlib.pyplot as plt
import numpy as np


def count_values(data):
    unique_values, counts = np.unique(data, return_counts=True)
    return dict(zip(unique_values, counts)) #slownik - klucz (unikalne wartosci) - values (liczby wystapien)


def plot_bar(Y_test, Y_pred):
    actual_counts = count_values(Y_test)
    predicted_counts = count_values(Y_pred)
    all_labels = set(list(actual_counts.keys()) + list(predicted_counts.keys()))

    plt.figure(figsize=(10, 5))

    plt.bar([label + " (Rzeczywiste)" for label in all_labels], [actual_counts.get(label, 0) for label in all_labels], color='blue', alpha=0.7, label='Rzeczywiste')

    plt.bar([label + " (Przewidywane)" for label in all_labels], [predicted_counts.get(label, 0) for label in all_labels], color='orange', alpha=0.7, label='Przewidywane')

    plt.xlabel('Wartość G3')
    plt.ylabel('Liczba wystąpień')
    plt.title('Liczba wystąpień rzeczywistych i przewidywanych wartości G3')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.show()