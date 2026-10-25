import copy

import numpy as np
from functions import (sigmoid, sigmoid_derivative, mean_squared_error,
                       mean_squared_error_derivative)
from layer import Layer
import matplotlib.pyplot as plt
from paths import PATH_STATS, PATH_TRAIN_EFFECTS
import logging
import os


def _create_logger(name, path):
    """Osobny logger z wlasnym plikiem - basicConfig dziala tylko raz na proces."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


stats_logger = _create_logger("statystyki", PATH_STATS)
effects_logger = _create_logger("skutecznosc", PATH_TRAIN_EFFECTS)


class Network:
    def __init__(self, neurons_in_particular_layers: list, bias, activations=None):
        """bias: wartosc wspolna albo lista per warstwa.
        activations: lista par (aktywacja, pochodna), domyslnie sigmoida.
        """
        self.num_layers = len(neurons_in_particular_layers)
        self.neurons_in_particular_layers = neurons_in_particular_layers
        self.input_num = neurons_in_particular_layers[0]
        self.layer_list = []
        self.bias = bias
        inputs_layer = self.input_num

        layer_count = self.num_layers - 1
        self.layer_biases = (list(bias) if isinstance(bias, (list, tuple))
                             else [bias] * layer_count)
        if activations is None:
            activations = [(sigmoid, sigmoid_derivative)] * layer_count

        for i in range(1, self.num_layers):
            activation, derivative = activations[i - 1]
            self.layer_list.append(Layer(inputs_layer, neurons_in_particular_layers[i],
                                         self.layer_biases[i - 1], activation, derivative))
            inputs_layer = neurons_in_particular_layers[i]

        derivatives = [np.zeros((self.input_num, self.layer_list[0].neuron_num))]
        derivatives.extend([np.zeros((layer.neuron_num, self.layer_list[i + 1].neuron_num)) for i, layer in
                            enumerate(self.layer_list[:-1])])
        self.derivatives = derivatives
        activations = [np.zeros(self.input_num)]  # liczba aktywacji taka sama
        activations.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.activations = activations

        # Bufory wag startuja z rzeczywistymi wagami neuronow. Wczesniej byly zerami,
        # wiec w pierwszym kroku momentum diff = old_weights - weights = -W0 i zamiast
        # W1 = W0 - lr*grad wychodzilo W1 = (1 + momentum) * W0 - lr*grad, czyli skok
        # przeskalowujacy wszystkie wagi sieci (dla momentum=0.9 az 1.9-krotnie).
        self.weights = [layer.get_weights() for layer in self.layer_list]
        self.old_weights = [w.copy() for w in self.weights]

        biases = []
        biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.biases = biases

        derivatives_biases = []
        derivatives_biases.extend([np.zeros(layer.neuron_num) for layer in self.layer_list])
        self.derivatives_biases = derivatives_biases

        # statystyki normalizacji cech - wypelnia je fit_normalization(),
        # None oznacza dane uzywane bez skalowania (np. autoasocjacja)
        self.feature_mean = None
        self.feature_std = None

    def fit_normalization(self, data):
        """Liczy srednia i odchylenie cech na danych TRENINGOWYCH i zapamietuje je.

        Statystyki licza sie wylacznie na zbiorze treningowym - policzenie ich
        na tescie byloby wyciekiem informacji. Zyja na obiekcie sieci, wiec
        wedruja razem z nia przy zapisie do pliku i test nie moze sie z nimi
        rozjechac.
        """
        self.feature_mean = data[:, :self.input_num].mean(axis=0)
        self.feature_std = data[:, :self.input_num].std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # cecha stala - nie dziel przez zero
        return self.normalize(data)

    def disable_normalization(self):
        """Kasuje zapamietane statystyki cech.

        Bez tego siec uczona wczesniej na skalowanym zbiorze dalej skalowalaby
        nowy zbior srednia i odchyleniem poprzedniego.
        """
        self.feature_mean = None
        self.feature_std = None

    def normalize(self, data):
        """Skaluje cechy wedlug zapamietanych statystyk (z-score).

        Siec bez statystyk zwraca dane nietkniete. getattr obsluguje modele
        zapisane do pliku przed wprowadzeniem tych pol.
        """
        if getattr(self, "feature_mean", None) is None:
            return data
        data = data.copy()
        data[:, :self.input_num] = (data[:, :self.input_num] - self.feature_mean) / self.feature_std
        return data

    def forward(self, input_vector):
        self.activations[0] = np.array(input_vector)
        next_activation = self.activations[0]
        for x in range(len(self.layer_list)):
            next_activation = self.layer_list[x].forward(next_activation)
            self.activations[x + 1] = next_activation
            self.weights[x] = self.layer_list[x].get_weights()
            self.biases[x] = self.layer_list[x].get_biases()

    def back_propagation(self, error):
        for x in reversed(range(len(self.derivatives))):
            activation = self.activations[x + 1]
            delta = error * self.layer_list[x].derivative(activation)
            delta_resh = delta.reshape(delta.shape[0], -1).T
            curr_act = self.activations[x]
            curr_act_resh = curr_act.reshape(curr_act.shape[0], -1)
            self.derivatives[x] = np.dot(curr_act_resh, delta_resh)

            error = np.dot(delta, self.weights[x])

            self.derivatives_biases[x] = delta

    def update_weights(self, learning_rate, momentum):
        for i in range(len(self.weights)):
            derivatives = self.derivatives[i]
            if momentum != 0:

                remember = copy.deepcopy(self.weights[i])
                diff = self.old_weights[i] - self.weights[i]
                self.weights[i] -= derivatives.T * learning_rate + (momentum * diff)

                self.old_weights[i] = remember

                #print("sdasd")
                # current_weights = self.weights[i]
                #
                # dif = self.weights[i] - self.old_weights[i]
                # subtractCalc = (derivatives.T * learning_rate) + (momentum * dif)
                # calculatedNewWeights = self.weights[i] - subtractCalc
                #
                # self.weights[
                #     i] = calculatedNewWeights  # self.weights[i] - derivatives.T * learning_rate + (momentum * dif)

                # self.old_weights[i] = current_weights

                # dif = self.weights[i] - self.old_weights[i]
                # a = self.weights[i]
                # b = self.old_weights[i]
                #print("alsihdkaushd")
                # aoishpdiuas
            else:
                self.weights[i] -= derivatives.T * learning_rate
        self.update_neurons_weights()

    def update_biases(self, learning_rate):
        for i in range(len(self.biases)):
            if not self.layer_biases[i]:
                continue
            derivatives_biases = self.derivatives_biases[i]
            self.biases[i] -= derivatives_biases * learning_rate
        self.update_neurons_biases()

    def update_neurons_weights(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_weights(self.weights[x])

    def update_neurons_biases(self):
        for x in range(len(self.layer_list)):
            self.layer_list[x].update_biases(self.biases[x])

    def prepare_samples(self, data, is_last_column_class):
        """Sprowadza dane do postaci [wejscia | wzorzec] i sprawdza zgodnosc wymiarow.

        Wczesniej train() i test() robily to osobno, mieszajac dwa sposoby
        indeksowania: cechy brane jako row[:-1], a numer klasy jako
        row[self.input_num]. Dla irysow obie formuly wskazuja te sama kolumne,
        ale dla dowolnego innego zbioru sie rozjezdzaja i uczenie konczylo sie
        nieobsluzonym bledem broadcastingu. Teraz indeksowanie jest jedno,
        a niezgodnosc ksztaltu daje czytelny komunikat przed uczeniem.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        num_outputs = self.neurons_in_particular_layers[-1]

        if not is_last_column_class:
            expected = self.input_num + num_outputs
            if data.shape[1] != expected:
                raise ValueError(
                    f"Siec ma {self.input_num} wejsc i {num_outputs} wyjsc, wiec zbior "
                    f"powinien miec {expected} kolumn, a ma {data.shape[1]}.")
            return data.copy()

        expected = self.input_num + 1
        if data.shape[1] != expected:
            raise ValueError(
                f"Siec ma {self.input_num} wejsc, wiec zbior klasyfikacyjny powinien miec "
                f"{expected} kolumn ({self.input_num} cech + kolumna klasy), "
                f"a ma {data.shape[1]}.")

        prepared = np.zeros((len(data), self.input_num + num_outputs))
        prepared[:, :self.input_num] = data[:, :self.input_num]
        for i, label in enumerate(data[:, self.input_num]):
            if label != int(label) or not 0 <= int(label) < num_outputs:
                raise ValueError(
                    f"Wiersz {i + 1}: numer klasy {label} jest poza zakresem "
                    f"0..{num_outputs - 1}, bo warstwa wyjsciowa ma {num_outputs} neuronow.")
            prepared[i, self.input_num + int(label)] = 1.0
        return prepared

    def train(self, input_data, epochs, shuffle, learning_rate, jump, given_error, momentum, is_last_column_class):
        mistakes = []
        x_values = []
        avg_error = None
        reached_given_error = False
        # prepare_samples zwraca wlasna kopie, wiec shuffle ponizej nie miesza
        # tablicy nalezacej do wywolujacego
        input_data = self.prepare_samples(input_data, is_last_column_class)
        for i in range(epochs):
            if shuffle:
                np.random.shuffle(input_data)
            sum_errors = 0
            for sample in input_data:

                self.forward(sample[:self.input_num])
                target = sample[self.input_num:]
                error = mean_squared_error_derivative(target, self.activations[-1])
                self.back_propagation(error)

                self.update_weights(learning_rate, momentum)

                if any(self.layer_biases):
                    self.update_biases(learning_rate)

                sum_errors += mean_squared_error(target, self.activations[-1])
            avg_error = sum_errors / len(input_data)
            # epoke konczaca uczenie zapisujemy zawsze, zeby wykres siegal do konca
            if i % jump == 0 or avg_error <= given_error:
                stats_logger.info(f"{avg_error}")
                mistakes.append(avg_error)
                x_values.append(i)
            if avg_error <= given_error:
                print(avg_error)
                stats_logger.info(f"Uzyskano zadany poziom bledu w epoce {i}\n")
                stats_logger.info(f"blad wynosi: {avg_error}\n")
                reached_given_error = True
                break
        if not reached_given_error:
            stats_logger.info(f"Nie osiagnieto podanego bledu {given_error} po {epochs} epokach, "
                              f"koncowy blad: {avg_error}")
        # jeden wykres, rysowany z faktycznie zebranych punktow - przy przerwaniu
        # petli x_values jest krotsze niz arange(0, epochs, jump) i plot wysypywal sie
        # na "x and y must have same first dimension"
        # nowa figura na kazdy trening - bez tego matplotlib dorysowywal kolejne
        # przebiegi na te same osie i wykresy z roznych uczen sie nakladaly
        plt.figure()
        plt.plot(x_values, mistakes)
        # zakres osi Y z faktycznych bledow - zaszyte 0.3 bylo dobrane pod irysy
        # i przy innym zbiorze ucinalo poczatek krzywej uczenia
        top = max(mistakes) if mistakes else 0
        plt.ylim(0, top * 1.05 if top > 0 else 1.0)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title("Zalezność bledu od ilości epok")
        plt.xlabel("Liczba epok")
        plt.ylabel("Wartosc bledu")
        plt.show()

    @staticmethod
    def comparison_table(rows, nr_of_outputs, limit=None):
        """Buduje tabelke porownujaca wyjscia neuronow z wartosciami wzorcowymi.

        rows: lista krotek (nr, wyjscia, wzorzec, blad, klasa_przewidziana, klasa_wzorcowa)
        """

        def fmt_target(value):
            return f"{value:>7.0f}" if float(value).is_integer() else f"{value:>7.3f}"

        outputs_header = "".join(f"{'y' + str(i + 1):>9}" for i in range(nr_of_outputs))
        targets_header = "".join(f"{'t' + str(i + 1):>7}" for i in range(nr_of_outputs))
        header = (f"{'nr':>4} |{outputs_header} |{targets_header} | {'blad':>9} |"
                  f" {'pred':>4} | {'wzor':>4} | wynik")
        lines = ["", "Porownanie wyjsc sieci (y) z wzorcem (t)", header, "-" * len(header)]
        shown = rows if limit is None else rows[:limit]
        for nr, outputs, targets, error, predicted, expected in shown:
            outputs_text = "".join(f"{value:>9.5f}" for value in outputs)
            targets_text = "".join(fmt_target(value) for value in targets)
            verdict = "OK" if predicted == expected else "BLAD"
            lines.append(f"{nr:>4} |{outputs_text} |{targets_text} | {error:>9.5f} |"
                         f" {predicted:>4} | {expected:>4} | {verdict}")
        lines.append("-" * len(header))
        if limit is not None and len(rows) > limit:
            lines.append(f"... pominieto {len(rows) - limit} wierszy "
                         f"(pelna tabela jest w pliku z wynikami)")
        return "\n".join(lines)

    def test(self, test_data, is_last_column_class, print_limit=30):
        stats_logger.info("\n\n\nTestowanie\n\n")
        nr_of_outputs = self.neurons_in_particular_layers[-1]
        test_data = self.prepare_samples(test_data, is_last_column_class)

        correctly_guessed = 0
        incorrectly_guessed = 0
        comparison_rows = []
        confusion_matrix = np.zeros((nr_of_outputs, nr_of_outputs))
        for i, sample in enumerate(test_data):
            target = sample[self.input_num:]
            sample_input = sample[:self.input_num]
            self.forward(sample_input)
            output = self.activations[-1]
            predicted_class = np.argmax(output)
            target_class = int(np.argmax(target))
            error = mean_squared_error(target, self.activations[-1])
            stats_logger.info(
                f"activations: {['{:.5f}'.format(a) for a in output]}, target: {target}, error: {'{:.5f}'.format(error)}")
            comparison_rows.append((i + 1, np.array(output, dtype=float), np.array(target, dtype=float),
                                    error, int(predicted_class), target_class))
            confusion_matrix[target_class, predicted_class] += 1

        effects_logger.info(self.comparison_table(comparison_rows, nr_of_outputs))
        print(self.comparison_table(comparison_rows, nr_of_outputs, limit=print_limit))
        confusion_string = ""
        for i in range(nr_of_outputs):
            confusion_string += f"class: {i}\n"
            tp = confusion_matrix[i, i]
            tn = 0
            fp = 0
            fn = 0
            for j in range(nr_of_outputs):
                for k in range(nr_of_outputs):
                    if k != i and j != i:
                        tn += confusion_matrix[j, k]

            for j in range(nr_of_outputs):
                if j != i:
                    fp += confusion_matrix[j, i]
                    fn += confusion_matrix[i, j]

            confusion_string += f"TP: {tp} TN: {tn} FP: {fp} FN: {fn}\n"
            if (tp + fp != 0):
                precision = tp / (tp + fp)
            else:
                precision = 0
            if (tp != 0):
                recall = tp / (tp + fn)
            else:
                recall = 0
            if (precision + recall != 0):
                f_measure = 2 * (precision * recall) / (precision + recall)
            else:
                f_measure = 0
            confusion_string += f"Precision: {precision} Recall: {recall} F-measure: {f_measure} \n\n"
            # stats_logger.info(confusion_string)
            correctly_guessed += tp
            incorrectly_guessed += fp
        classified = correctly_guessed + incorrectly_guessed
        accuracy = correctly_guessed / classified * 100 if classified else 0.0
        print(f"Dobrze zakwalifikowano {round(accuracy, 5)} % ({int(correctly_guessed)}/{int(classified)})")
        print(confusion_matrix)
        effects_logger.info(confusion_string)
        effects_logger.info(f"\n correctly classified: {correctly_guessed}\n")
        effects_logger.info(f"\n incorrectly classified: {incorrectly_guessed}\n")
