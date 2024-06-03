import matplotlib.pyplot as plt
import numpy as np


def count_values(data):
    unique_values, counts = np.unique(data, return_counts=True)
    return dict(zip(unique_values, counts)) #slownik - klucz (unikalne wartosci) - values (liczby wystapien)


def plot_bar(Y_test, Y_pred, grade, subject):
    actual_counts = count_values(Y_test)
    predicted_counts = count_values(Y_pred)
    all_labels = set(list(actual_counts.keys()) + list(predicted_counts.keys()))
    sorted_labels = sorted(all_labels, key=lambda x: actual_counts.get(x, 0))
    sorted_actual_counts = [actual_counts.get(label, 0) for label in sorted_labels]
    sorted_predicted_counts = [predicted_counts.get(label, 0) for label in sorted_labels]

    labels = list(sorted_labels)
    n = len(labels)
    bar_width = 0.4 

    plt.figure(figsize=(10, 5))

    
    plt.bar(np.arange(n), sorted_actual_counts, color='blue', width=bar_width, alpha=0.7, label='Rzeczywiste')

    plt.bar(np.arange(n) + bar_width, sorted_predicted_counts, color='orange', width=bar_width, alpha=0.7, label='Przewidywane')

    plt.xlabel(f'Wartość {grade}')
    plt.ylabel('Liczba wystąpień')
    plt.title(f'Liczba wystąpień rzeczywistych i przewidywanych wartości {grade}  dla toku ksztalcenia {subject}')
    plt.xticks(np.arange(n) + bar_width / 2, labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    plt.show()