"""Interaktywne CLI do uczenia i testowania sieci MLP na dowolnym zbiorze CSV."""

import os
import sys
import textwrap

from NeuronNetwork import Network
from dataset import (DatasetError, FORMAT_CLASS, FORMAT_IO, FORMAT_NAMES,
                     from_class_file, from_io_file, split)
from file_service import save, load
from paths import PATH_ASOCJACJA, PATH_SAVED, PATH_TEST_DATA, PATH_TRAIN_DATA

EFFECTIVE_STEP_LIMIT = 10.0


# ---------------------------------------------------------------- wejscie

def _read_line(prompt):
    """Czyta linie. Ctrl-D (EOF) konczy program zamiast rzucac EOFError."""
    try:
        return input(prompt)
    except EOFError:
        print("\nKoniec wejscia - zamykam program.")
        sys.exit()


def _show_help(help_text):
    print()
    print(textwrap.indent(help_text.strip(), "  "))
    print()


def ask_text(prompt, help_text=None, default=None):
    """Pyta o tekst. '?' pokazuje podpowiedz, pusta linia bierze wartosc domyslna."""
    full = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    while True:
        value = _read_line(full).strip()
        if value == "?":
            _show_help(help_text or "Brak dodatkowej podpowiedzi dla tego pytania.")
            continue
        if not value and default is not None:
            return str(default)
        if not value:
            print("Wpisz wartosc albo '?' po podpowiedz.")
            continue
        return value


def ask_int(prompt, help_text=None, default=None, minimum=None, maximum=None):
    while True:
        raw = ask_text(prompt, help_text, default)
        try:
            value = int(raw)
        except ValueError:
            print("Podaj liczbe calkowita (albo '?' po podpowiedz).")
            continue
        if minimum is not None and value < minimum:
            print(f"Wartosc musi byc nie mniejsza niz {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Wartosc musi byc nie wieksza niz {maximum}.")
            continue
        return value


def ask_float(prompt, help_text=None, default=None, minimum=None, maximum=None,
              exclusive_min=False, exclusive_max=False):
    while True:
        raw = ask_text(prompt, help_text, default)
        try:
            value = float(raw)
        except ValueError:
            print("Podaj liczbe (albo '?' po podpowiedz).")
            continue
        if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
            print(f"Wartosc musi byc {'wieksza niz' if exclusive_min else 'nie mniejsza niz'} {minimum}.")
            continue
        if maximum is not None and (value >= maximum if exclusive_max else value > maximum):
            print(f"Wartosc musi byc {'mniejsza niz' if exclusive_max else 'nie wieksza niz'} {maximum}.")
            continue
        return value


def ask_yes_no(prompt, help_text=None, default=None):
    default_text = None if default is None else ("t" if default else "n")
    while True:
        value = ask_text(f"{prompt} [t/n]", help_text, default_text).strip().lower()
        if value in ("t", "tak", "y", "yes"):
            return True
        if value in ("n", "nie", "no"):
            return False
        print("Wpisz 't' albo 'n'.")


def _path_completer(text, state):
    """Uzupelnianie sciezek tabulatorem - w CLI o plikach to podstawowa wygoda."""
    directory, separator, prefix = text.rpartition(os.sep)
    listing_dir = directory + separator if separator else "."
    try:
        names = sorted(os.listdir(listing_dir))
    except OSError:
        return None
    head = text[:len(text) - len(prefix)]
    matches = [head + name + (os.sep if os.path.isdir(os.path.join(listing_dir, name)) else "")
               for name in names if name.startswith(prefix)]
    return matches[state] if state < len(matches) else None


def ask_path(prompt, help_text=None, default=None):
    """Pyta o sciezke do pliku, z uzupelnianiem tabulatorem jesli readline jest dostepny."""
    try:
        import readline
    except ImportError:
        return ask_text(prompt, help_text, default)
    previous = readline.get_completer()
    readline.set_completer(_path_completer)
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        return os.path.expanduser(ask_text(prompt, help_text, default))
    finally:
        readline.set_completer(previous)


# ---------------------------------------------------------------- pomoc

HELP_GENERAL = """
Program uczy wielowarstwowa siec neuronowa (MLP) metoda wstecznej propagacji
bledu z momentum. Dziala na dowolnym zbiorze danych w formacie CSV.

Typowa kolejnosc pracy:
  1. [6] wybierz zbior treningowy (i testowy) z pliku
  2. [1] stworz siec - program zaproponuje strukture pasujaca do zbioru
  3. [7] ustaw parametry uczenia (wspolczynnik uczenia, momentum, epoki)
  4. [4] trenuj, potem [5] testuj
  5. [2] zapisz nauczona siec do pliku, zeby wrocic do niej pozniej

W kazdym pytaniu mozesz wpisac '?', zeby zobaczyc podpowiedz, albo wcisnac
Enter, zeby przyjac wartosc domyslna podana w nawiasie kwadratowym.
Przy pytaniach o plik dziala uzupelnianie sciezki tabulatorem.

Obslugiwane formaty plikow CSV (bez naglowka albo z jednym naglowkiem,
ktory program sam pominie):

  Format A - klasyfikacja
    Kolumny cech, a ostatnia kolumna to numer klasy: 0, 1, 2, ...
    Przyklad (irysy): 5.1,3.5,1.4,0.2,0
    Liczba wejsc sieci = liczba kolumn cech, liczba wyjsc = liczba klas.

  Format B - wejscia i wyjscia
    N pierwszych kolumn to wejscia, wszystkie pozostale to wzorzec wyjscia.
    Podajesz N samodzielnie.
    Przyklad (autoasocjacja 4-2-4): 1,0,0,0,1,0,0,0
"""

HELP_MENU = {
    "1": """
[1] Stworz siec
    Budujesz nowa siec: liczba neuronow wejsciowych, warstwy ukryte,
    liczba neuronow wyjsciowych i obecnosc biasu.
    Jesli masz juz wybrany zbior treningowy, program zaproponuje liczbe
    wejsc i wyjsc pasujaca do niego - wystarczy zatwierdzic Enterem.
    Wagi startowe sa losowane z przedzialu [-1, 1].
    Uwaga: tworzenie nowej sieci kasuje dotychczas nauczona siec.
""",
    "2": """
[2] Zapisz siec do pliku
    Zapisuje cala siec (wagi, biasy, strukture i statystyki normalizacji)
    przez pickle. Statystyki normalizacji jada razem z siecia, wiec po
    wczytaniu test skaluje dane dokladnie tak samo jak uczenie.
""",
    "3": """
[3] Wczytaj siec z pliku
    Odczytuje siec zapisana wczesniej opcja [2]. Struktura i ustawienie
    biasu sa brane z pliku, wiec nadpisuja biezace ustawienia.
""",
    "4": """
[4] Trenuj siec
    Uczy siec na wybranym zbiorze treningowym parametrami z [7].
    Uczenie jest na-probkowe (online): wagi sa poprawiane po kazdej probce.
    Jesli siec o tej samej strukturze juz istnieje, program zapyta, czy
    kontynuowac jej uczenie, czy zaczac od nowa z losowymi wagami.
    Na koncu pokazuje wykres bledu w funkcji liczby epok.
""",
    "5": """
[5] Testuj siec
    Przepuszcza zbior testowy przez siec bez zmiany wag i pokazuje:
    tabele porownujaca wyjscia neuronow z wzorcem, macierz pomylek oraz
    precyzje, czulosc i F-miare dla kazdej klasy.
    Pelne wyniki trafiaja do plikow w katalogu files/.
""",
    "6": """
[6] Zbiory danych
    Podmenu wyboru danych: zbior treningowy, zbior testowy, podzial
    jednego pliku na obie czesci, zbiory wbudowane oraz przelacznik
    normalizacji cech.
""",
    "7": """
[7] Parametry uczenia
    Podmenu z parametrami: wspolczynnik uczenia, momentum, liczba epok,
    prog bledu konczacy uczenie, co ile epok zapisywac blad i czy tasowac
    probki miedzy epokami.
""",
}

HELP_DATA_MENU = """
Zbiory danych

  [1] Zbior treningowy z pliku - dane, na ktorych siec sie uczy.
  [2] Zbior testowy z pliku - dane do oceny, nieuzywane w uczeniu.
  [3] Podziel jeden plik - gdy masz jeden plik zamiast dwoch. Program
      losowo dzieli probki w zadanej proporcji. Ziarno losowania mozesz
      podac, zeby podzial dal sie powtorzyc.
  [4] Zbiory wbudowane - gotowe irysy (klasyfikacja 4 -> 3) albo
      autoasocjacja (4 wejscia -> 4 wyjscia).
  [5] Normalizacja cech - skalowanie z-score (odjecie sredniej, podzielenie
      przez odchylenie). Srednia i odchylenie licza sie WYLACZNIE na zbiorze
      treningowym i sa potem stosowane do testowego, zeby informacja z testu
      nie wyciekla do uczenia. Warto wlaczyc, gdy cechy maja rozne jednostki
      lub rzedy wielkosci (np. irysy). Nie ma sensu dla danych zero-jedynkowych
      (np. autoasocjacja).
"""

HELP_PARAMS = {
    "lr": """
Wspolczynnik uczenia (learning rate)
    Dlugosc kroku poprawki wag. Za maly - uczenie trwa wiecznie. Za duzy -
    blad skacze i siec sie nie zbiega. Rozsadny start to 0.01 - 0.3.
""",
    "momentum": """
Momentum
    Czesc poprzedniej poprawki dodawana do biezacej. Wygladza uczenie i
    pomaga wyjsc z plytkich minimow lokalnych. Zakres [0, 1). 0 wylacza
    momentum, typowe wartosci to 0.6 - 0.9.
    Uwaga: poprawki sumuja sie geometrycznie, wiec realna dlugosc kroku to
    learning_rate / (1 - momentum), a nie samo learning_rate.
""",
    "epochs": """
Liczba epok
    Ile razy siec zobaczy caly zbior treningowy. Uczenie konczy sie po tylu
    epokach albo wczesniej, jesli blad spadnie do progu z nastepnego pytania.
""",
    "error": """
Prog bledu konczacy uczenie
    Gdy sredni blad epoki spadnie do tej wartosci, uczenie sie zatrzymuje.
    Wpisz 0, zeby zawsze wykonac pelna liczbe epok (blad nigdy nie osiaga zera).
""",
    "jump": """
Co ile epok zapisywac blad
    Krok probkowania wykresu i pliku ze statystykami. 1 zapisuje kazda epoke,
    wieksze wartosci daja rzadsze punkty i mniejszy plik.
""",
    "shuffle": """
Tasowanie probek
    Losowa zmiana kolejnosci probek przed kazda epoka. Zwykle warto, bo
    przeciwdziala uczeniu sie kolejnosci danych zamiast samych wzorcow.
""",
}

HELP_FORMAT = f"""
Format pliku:
  [1] {FORMAT_NAMES[FORMAT_CLASS]}
      Przyklad wiersza: 5.1,3.5,1.4,0.2,0
  [2] {FORMAT_NAMES[FORMAT_IO]}
      Przyklad wiersza: 1,0,0,0,1,0,0,0  (przy 4 wejsciach)
"""


# ---------------------------------------------------------------- stan

class Session:
    """Caly stan CLI w jednym miejscu: siec, zbiory i parametry uczenia."""

    def __init__(self):
        self.network = None
        self.structure = [4, 10, 10, 3]
        self.bias = True
        self.train_set = None
        self.test_set = None
        self.normalize = True
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.epochs = 150
        self.given_error = 0.0
        self.jump = 2
        self.shuffle = True

    def status(self):
        lines = []
        if self.network is None:
            lines.append(f"  Siec:       brak (struktura docelowa {self.structure}, "
                         f"bias={'tak' if self.bias else 'nie'})")
        else:
            lines.append(f"  Siec:       {self.network.neurons_in_particular_layers}, "
                         f"bias={'tak' if self.network.bias else 'nie'}")
        lines.append(f"  Treningowy: {self.train_set.summary() if self.train_set else 'nie wybrano'}")
        lines.append(f"  Testowy:    {self.test_set.summary() if self.test_set else 'nie wybrano'}")
        lines.append(f"  Uczenie:    lr={self.learning_rate}, momentum={self.momentum}, "
                     f"epoki={self.epochs}, prog bledu={self.given_error}, "
                     f"zapis co {self.jump}, tasowanie={'tak' if self.shuffle else 'nie'}, "
                     f"normalizacja={'tak' if self.normalize else 'nie'}")
        return "\n".join(lines)


# ---------------------------------------------------------------- zbiory

def choose_format(default="1"):
    choice = ask_text("Format pliku [1] klasyfikacja / [2] wejscia+wyjscia",
                      HELP_FORMAT, default)
    return FORMAT_CLASS if choice.strip() == "1" else FORMAT_IO


def load_dataset(prompt, default_path=None, expected_format=None,
                 expected_inputs=None, expected_outputs=None):
    """Pyta o plik i format, po czym wczytuje zbior. None przy bledzie."""
    path = ask_path(prompt, HELP_GENERAL, default_path)
    data_format = expected_format or choose_format()
    try:
        if data_format == FORMAT_CLASS:
            dataset = from_class_file(path, output_num=expected_outputs)
        else:
            inputs = expected_inputs or ask_int(
                "Ile pierwszych kolumn to wejscia",
                "Pozostale kolumny zostana potraktowane jako wzorzec wyjscia.",
                minimum=1)
            dataset = from_io_file(path, inputs)
    except DatasetError as error:
        print(f"\nBlad: {error}\n")
        return None

    print(f"\nWczytano {dataset.summary()}")
    if dataset.header_skipped:
        print("  Pominieto pierwszy wiersz - wyglada na naglowek.")
    if dataset.format == FORMAT_CLASS:
        counts = dataset.class_counts()
        print("  Liczebnosc klas: " + ", ".join(f"{c}:{n}" for c, n in enumerate(counts)))
        if dataset.missing_classes:
            print(f"  Uwaga: brak probek dla klas {dataset.missing_classes}.")
    print()
    return dataset


def data_menu(session):
    while True:
        print("\n" + "=" * 62)
        print("ZBIORY DANYCH")
        print(session.status())
        print("-" * 62)
        print("  [1] Zbior treningowy z pliku")
        print("  [2] Zbior testowy z pliku")
        print("  [3] Podziel jeden plik na treningowy i testowy")
        print("  [4] Zbiory wbudowane (irysy / autoasocjacja)")
        print(f"  [5] Normalizacja cech: {'tak' if session.normalize else 'nie'} (przelacz)")
        print("  [?] Pomoc            [0] Powrot")
        choice = ask_text("Wybor", HELP_DATA_MENU)

        if choice == "0":
            return
        if choice == "1":
            dataset = load_dataset("Plik ze zbiorem treningowym")
            if dataset:
                session.train_set = dataset
                session.test_set = None
                session.normalize = dataset.format == FORMAT_CLASS
                print(f"Normalizacja ustawiona na {'tak' if session.normalize else 'nie'} "
                      f"(mozesz zmienic opcja [5]).")
                offer_structure(session, dataset)
        elif choice == "2":
            if session.train_set is None:
                print("\nNajpierw wybierz zbior treningowy - od niego brany jest format.\n")
                continue
            dataset = load_dataset("Plik ze zbiorem testowym",
                                   expected_format=session.train_set.format,
                                   expected_inputs=session.train_set.input_num,
                                   expected_outputs=session.train_set.output_num)
            if dataset:
                session.test_set = dataset
        elif choice == "3":
            dataset = load_dataset("Plik do podzialu")
            if not dataset:
                continue
            ratio = ask_float("Jaka czesc ma trafic do zbioru testowego",
                              "Ulamek z przedzialu (0, 1). 0.2 oznacza 20 % probek na test.",
                              default=0.2, minimum=0, maximum=1,
                              exclusive_min=True, exclusive_max=True)
            seed_text = ask_text("Ziarno losowania (pusto = losowe)",
                                 "Ta sama liczba daje zawsze ten sam podzial, co pozwala "
                                 "powtorzyc eksperyment.", default="-")
            seed = None if seed_text.strip() in ("", "-") else int(seed_text)
            try:
                train_part, test_part = split(dataset, ratio, seed)
            except DatasetError as error:
                print(f"\nBlad: {error}\n")
                continue
            session.train_set, session.test_set = train_part, test_part
            session.normalize = dataset.format == FORMAT_CLASS
            print(f"\nPodzielono: {train_part.sample_num} probek treningowych, "
                  f"{test_part.sample_num} testowych.")
            for part in (train_part, test_part):
                if part.missing_classes:
                    print(f"  Uwaga: w czesci '{part.name}' brak klas {part.missing_classes}.")
            offer_structure(session, train_part)
        elif choice == "4":
            builtin_menu(session)
        elif choice == "5":
            session.normalize = not session.normalize
            print(f"Normalizacja cech: {'wlaczona' if session.normalize else 'wylaczona'}.")
        elif choice != "?":
            print("Wybierz opcje z listy.")


def builtin_menu(session):
    print("\n  [1] Irysy (klasyfikacja, 4 cechy -> 3 klasy, gotowy podzial train/test)")
    print("  [2] Autoasocjacja (4 wejscia -> 4 wyjscia, ten sam plik do uczenia i testu)")
    print("  [0] Powrot")
    choice = ask_text("Wybor", "Zbiory dolaczone do projektu w katalogu data/.", "1")
    try:
        if choice == "1":
            session.train_set = from_class_file(PATH_TRAIN_DATA)
            session.test_set = from_class_file(
                PATH_TEST_DATA, output_num=session.train_set.output_num)
            session.normalize = True
        elif choice == "2":
            session.train_set = from_io_file(PATH_ASOCJACJA, 4)
            session.test_set = from_io_file(PATH_ASOCJACJA, 4)
            session.normalize = False
        else:
            return
    except DatasetError as error:
        print(f"\nBlad: {error}\n")
        return
    print(f"\nTreningowy: {session.train_set.summary()}")
    print(f"Testowy:    {session.test_set.summary()}")
    offer_structure(session, session.train_set)


def offer_structure(session, dataset):
    """Proponuje strukture sieci pasujaca wymiarami do wybranego zbioru."""
    suggested = dataset.suggested_structure() if dataset.format == FORMAT_CLASS \
        else [dataset.input_num, max(2, dataset.input_num // 2), dataset.output_num]
    current = session.network.neurons_in_particular_layers if session.network else session.structure
    if current[0] == dataset.input_num and current[-1] == dataset.output_num:
        return
    print(f"\nObecna siec {current} nie pasuje do zbioru "
          f"({dataset.input_num} wejsc, {dataset.output_num} wyjsc).")
    if ask_yes_no(f"Ustawic strukture {suggested}?",
                  "Mozesz tez zbudowac wlasna strukture opcja [1] w menu glownym. "
                  "Liczba neuronow wejsciowych i wyjsciowych musi zgadzac sie ze zbiorem, "
                  "warstwy ukryte sa dowolne.", default=True):
        session.structure = suggested
        session.network = None
        print(f"Struktura ustawiona na {suggested}. Siec powstanie przy treningu [4] "
              f"albo recznie opcja [1].")


# ---------------------------------------------------------------- parametry

def params_menu(session):
    editors = {
        "1": ("learning_rate", lambda: ask_float(
            "Wspolczynnik uczenia", HELP_PARAMS["lr"],
            session.learning_rate, minimum=0, exclusive_min=True)),
        "2": ("momentum", lambda: ask_float(
            "Momentum", HELP_PARAMS["momentum"],
            session.momentum, minimum=0, maximum=1, exclusive_max=True)),
        "3": ("epochs", lambda: ask_int(
            "Liczba epok", HELP_PARAMS["epochs"], session.epochs, minimum=1)),
        "4": ("given_error", lambda: ask_float(
            "Prog bledu konczacy uczenie", HELP_PARAMS["error"],
            session.given_error, minimum=0)),
        "5": ("jump", lambda: ask_int(
            "Co ile epok zapisywac blad", HELP_PARAMS["jump"], session.jump, minimum=1)),
        "6": ("shuffle", lambda: ask_yes_no(
            "Tasowac probki miedzy epokami?", HELP_PARAMS["shuffle"], session.shuffle)),
    }
    while True:
        print("\n" + "=" * 62)
        print("PARAMETRY UCZENIA")
        print("-" * 62)
        print(f"  [1] Wspolczynnik uczenia      = {session.learning_rate}")
        print(f"  [2] Momentum                  = {session.momentum}")
        print(f"  [3] Liczba epok               = {session.epochs}")
        print(f"  [4] Prog bledu konczacy       = {session.given_error}")
        print(f"  [5] Zapis bledu co ile epok   = {session.jump}")
        print(f"  [6] Tasowanie probek          = {'tak' if session.shuffle else 'nie'}")
        print("  [7] Ustaw wszystko po kolei")
        print("  [0] Powrot")
        choice = ask_text("Wybor", "Wpisz numer parametru, zeby go zmienic. "
                                   "Przy kazdym pytaniu '?' pokazuje opis parametru.")
        if choice == "0":
            return
        if choice == "7":
            for key in ("1", "2", "3", "4", "5", "6"):
                field, editor = editors[key]
                setattr(session, field, editor())
        elif choice in editors:
            field, editor = editors[choice]
            setattr(session, field, editor())
        elif choice != "?":
            print("Wybierz opcje z listy.")


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
    print("Przy momentum kolejne poprawki sumuja sie geometrycznie, wiec realna")
    print("dlugosc kroku to lr/(1-momentum), a nie samo learning_rate.")
    print(f"Zejscie do granicy {EFFECTIVE_STEP_LIMIT:.0f} wymaga przy momentum={momentum} "
          f"wartosci learning_rate={EFFECTIVE_STEP_LIMIT * (1 - momentum):.3f} lub mniejszej.")
    return ask_yes_no("Kontynuowac mimo to?", default=False)


# ---------------------------------------------------------------- siec

def create_network(session):
    dataset = session.train_set
    default_inputs = dataset.input_num if dataset else session.structure[0]
    default_outputs = dataset.output_num if dataset else session.structure[-1]
    if dataset:
        print(f"\nZbior treningowy {dataset.summary()} wymaga {default_inputs} wejsc "
              f"i {default_outputs} wyjsc - wartosci domyslne sa z niego wziete.")

    inputs = ask_int("Liczba neuronow wejsciowych",
                     "Musi rownac sie liczbie kolumn wejsciowych w zbiorze.",
                     default_inputs, minimum=1)
    structure = [inputs]
    hidden_count = ask_int("Liczba warstw ukrytych",
                           "0 daje perceptron bez warstw ukrytych. Typowo 1 albo 2.",
                           max(0, len(session.structure) - 2), minimum=0)
    previous_hidden = session.structure[1:-1]
    for i in range(hidden_count):
        default_hidden = previous_hidden[i] if i < len(previous_hidden) else 10
        structure.append(ask_int(f"Liczba neuronow w {i + 1} warstwie ukrytej",
                                 "Wiecej neuronow to wieksza pojemnosc sieci, ale wolniejsze "
                                 "uczenie i wieksze ryzyko przeuczenia.",
                                 default_hidden, minimum=1))
    structure.append(ask_int("Liczba neuronow wyjsciowych",
                             "Dla klasyfikacji rowna liczbie klas, dla formatu "
                             "wejscia+wyjscia rowna liczbie kolumn wzorca.",
                             default_outputs, minimum=1))
    bias = ask_yes_no("Czy neurony maja miec bias?",
                      "Bias to dodatkowy, stale pobudzony wejsciowy wyraz. Pozwala "
                      "przesuwac funkcje aktywacji i zwykle warto go zostawic.",
                      session.bias)

    session.structure = structure
    session.bias = bias
    session.network = Network(structure, bias)
    print(f"Utworzono siec {structure}, bias={'tak' if bias else 'nie'}.")
    warn_if_mismatched(session)


def warn_if_mismatched(session, structure=None):
    """Sprawdza, czy struktura sieci zgadza sie wymiarami ze zbiorem treningowym.

    structure pozwala sprawdzic strukture PLANOWANA, zanim siec powstanie -
    inaczej trening najpierw pytalby o dotrenowanie, a dopiero potem okazywaloby
    sie, ze siec i tak nie pasuje do danych.
    """
    dataset = session.train_set
    if dataset is None:
        return True
    if structure is None:
        if session.network is None:
            return True
        structure = session.network.neurons_in_particular_layers
    problems = []
    if structure[0] != dataset.input_num:
        problems.append(f"siec ma {structure[0]} wejsc, a zbior {dataset.input_num}")
    if structure[-1] != dataset.output_num:
        problems.append(f"siec ma {structure[-1]} wyjsc, a zbior {dataset.output_num}")
    if problems:
        print("\nNiezgodnosc sieci ze zbiorem: " + "; ".join(problems) + ".")
        print("Zmien strukture opcja [1] albo wybierz inny zbior opcja [6].")
        return False
    return True


def prepare_network(session):
    """Zwraca siec do trenowania: istniejaca (dotrenowanie) albo swiezo utworzona."""
    existing = session.network
    if (existing is not None
            and existing.neurons_in_particular_layers == session.structure
            and existing.bias == session.bias):
        if ask_yes_no("Istnieje juz siec o tej strukturze. Kontynuowac jej uczenie?",
                      "'tak' dotrenowuje biezaca siec dalej. 'nie' tworzy nowa siec "
                      "z losowymi wagami, czyli zaczyna uczenie od zera.", default=True):
            print("Kontynuacja uczenia istniejacej sieci.")
            return existing
        print("Tworzenie nowej sieci.")
    return Network(session.structure, session.bias)


def do_train(session):
    if session.train_set is None:
        print("\nBrak zbioru treningowego - wybierz go opcja [6].\n")
        return
    if not warn_if_mismatched(session, session.structure):
        return
    if not confirm_effective_step(session.learning_rate, session.momentum):
        return
    session.network = prepare_network(session)

    data = session.train_set.data.copy()
    if session.normalize:
        data = session.network.fit_normalization(data)
    else:
        session.network.disable_normalization()

    try:
        session.network.train(
            data,
            epochs=session.epochs,
            shuffle=session.shuffle,
            learning_rate=session.learning_rate,
            jump=session.jump,
            given_error=session.given_error,
            momentum=session.momentum,
            is_last_column_class=session.train_set.format == FORMAT_CLASS)
    except ValueError as error:
        print(f"\nBlad podczas uczenia: {error}\n")


def do_test(session):
    if session.network is None:
        print("\nBrak sieci - wytrenuj ja [4] albo wczytaj z pliku [3].\n")
        return
    dataset = session.test_set or session.train_set
    if dataset is None:
        print("\nBrak zbioru testowego - wybierz go opcja [6].\n")
        return
    if session.test_set is None:
        print("Brak osobnego zbioru testowego - testuje na zbiorze treningowym.")
    try:
        session.network.test(session.network.normalize(dataset.data),
                             dataset.format == FORMAT_CLASS)
    except ValueError as error:
        print(f"\nBlad podczas testowania: {error}\n")


def do_save(session):
    if session.network is None:
        print("\nBrak sieci do zapisania - najpierw stworz ja [1] albo wytrenuj [4].\n")
        return
    path = ask_path("Plik docelowy", HELP_MENU["2"], PATH_SAVED)
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
        save(session.network, path)
    except OSError as error:
        print(f"\nNie udalo sie zapisac sieci: {error}\n")
        return
    print(f"Zapisano siec do {path}")


def do_load(session):
    path = ask_path("Plik z siecia", HELP_MENU["3"], PATH_SAVED)
    try:
        loaded = load(path)
    except (OSError, EOFError, ValueError) as error:
        print(f"\nNie udalo sie wczytac sieci: {error}\n")
        return
    if not isinstance(loaded, Network):
        print(f"\nPlik {path} nie zawiera sieci neuronowej.\n")
        return
    session.network = loaded
    session.structure = list(loaded.neurons_in_particular_layers)
    session.bias = loaded.bias
    session.normalize = getattr(loaded, "feature_mean", None) is not None
    print(f"Wczytano siec {session.structure}, bias={'tak' if session.bias else 'nie'}, "
          f"normalizacja={'tak' if session.normalize else 'nie'}")
    warn_if_mismatched(session)


# ---------------------------------------------------------------- menu

def main_menu(session):
    print("\n" + "=" * 62)
    print("SIEC NEURONOWA MLP - wsteczna propagacja bledu z momentum")
    print(session.status())
    print("-" * 62)
    print("  [1] Stworz siec              [5] Testuj siec")
    print("  [2] Zapisz siec do pliku     [6] Zbiory danych        >")
    print("  [3] Wczytaj siec z pliku     [7] Parametry uczenia    >")
    print("  [4] Trenuj siec              [?] Pomoc       [0] Koniec")
    print("-" * 62)
    print("  Podpowiedz: '?' w kazdym pytaniu, '?4' po opis opcji [4].")
    return ask_text("Wybor", HELP_GENERAL)


def main():
    session = Session()
    actions = {
        "1": create_network,
        "2": do_save,
        "3": do_load,
        "4": do_train,
        "5": do_test,
        "6": data_menu,
        "7": params_menu,
    }
    while True:
        choice = main_menu(session)
        if choice == "0":
            print("Koniec.")
            return
        if choice.startswith("?") and choice[1:] in HELP_MENU:
            _show_help(HELP_MENU[choice[1:]])
            continue
        if choice == "?":
            continue
        action = actions.get(choice)
        if action is None:
            print("\nNie ma takiej opcji. Wpisz '?' po liste mozliwosci.\n")
            continue
        action(session)


if __name__ == "__main__":
    main()
