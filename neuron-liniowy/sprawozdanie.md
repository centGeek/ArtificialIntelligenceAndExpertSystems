# Trening neuronu liniowego wieloma wzorcami — sprawozdanie

**Inteligentna Analiza Danych**, Instytut Informatyki Politechniki Łódzkiej

---

## 1. Zadanie i implementacja

Przedmiotem zadania jest **neuron liniowy**, czyli neuron bez funkcji aktywacji:

```
y = w · x = w₁x₁ + w₂x₂ + ... + w_N x_N
```

Trening wieloma wzorcami realizuje **Algorytm 3.2** (przyrostowa reguła delty). W każdej epoce
algorytm przechodzi po wszystkich *M* wzorcach i po każdym z nich natychmiast poprawia wagi:

```
y_j = w · x_j
w ← w + η (t_j − y_j) x_j
```

Implementacja: `linear_neuron.py` (klasa `LinearNeuron`, metoda `train`).
Eksperymenty: `experiments.py`.

Neuron nie ma biasu — liczba wag wynosi dokładnie *N*, zgodnie z treścią zadania. Dzięki temu
odpowiedniość z układem równań jest ścisła, bez dodatkowej niewiadomej.

### Kluczowa obserwacja o samym algorytmie

Każda poprawka wag jest wielokrotnością pewnego wzorca `x_j`. Zatem **cały przyrost wag
`w − w₀` leży w przestrzeni wierszowej macierzy X**, a składowa wag leżąca w jądrze `X` nigdy
nie jest modyfikowana — zostaje taka, jaka wypadła w losowaniu początkowym. Z tego jednego
faktu wynikają wszystkie zaobserwowane niżej prawidłowości. Formalnie granica treningu to:

```
w∞ = w₀ + X⁺ (t − X w₀)
```

## 2. Metodyka

Trening neuronu liniowego jest równoważny rozwiązywaniu układu *M* równań z *N* niewiadomymi:

```
X w = t,    X ma wymiar M × N
```

gdzie wierszami `X` są wzorce treningowe, a wagi neuronu są niewiadomymi.

Dla każdego z trzech przypadków:

- zbiór treningowy (`X` i `t`) losowany jest **raz** i używany przez wszystkie przebiegi,
- wagi początkowe losowane są **osobno dla każdego przebiegu** z rozkładu jednostajnego [−1, 1],
- wykonano **8 przebiegów** treningu,
- współczynnik uczenia dobierany jest automatycznie z danych jako `η = 0.2 / λmax(XᵀX)`,
  bo przy zbyt dużym η reguła delty rozbiega i porównywanie wag traci sens,
- `t` losowane jest **niezależnie** od `X`. To istotne: gdyby przyjąć `t = X w_prawdziwe`,
  układ byłby zgodny także dla *M > N* i najciekawszy przypadek by zniknął.

## 3. Wyniki

| przypadek | N | M | rząd X | rozrzut wag między przebiegami | błąd końcowy | granica najmn. kwadratów | skuteczny trening |
|---|---|---|---|---|---|---|---|
| N < M | 3 | 10 | 3 | **1.68e-10** | 4.93e-01 | 0.4929 | **NIE** |
| N = M | 5 | 5 | 5 | **5.91e-10** | 1.00e-14 | 0.0000 | **TAK** |
| N > M | 10 | 4 | 4 | **2.28e+00** | 9.97e-15 | 0.0000 | **TAK** |

Pełne tabele wag z każdego przebiegu: `wyniki/wyniki.md`.
Wykresy: `wyniki/zbieznosc_bledu.png`, `wyniki/wagi_koncowe.png`.

### Weryfikacja numeryczna przewidywań teoretycznych

| sprawdzenie | N < M | N = M | N > M |
|---|---|---|---|
| odchyłka od `w₀ + X⁺(t − X w₀)` | 5.42e-03 | 6.67e-07 | **1.35e-07** |
| `‖X (wᵢ − w₁)‖` — czy różnice leżą w jądrze X | 2.91e-10 | 1.24e-10 | **2.58e-07** |

Druga linijka jest najważniejsza dla przypadku *N > M*: wagi końcowe różnią się między
przebiegami o **2.28**, ale po przemnożeniu tych różnic przez `X` otrzymujemy praktycznie zero.
Oznacza to, że przebiegi różnią się **wyłącznie** składową leżącą w jądrze `X` — czyli w
kierunkach, których wzorce treningowe w ogóle nie widzą.

### Powtarzalność dla różnych losowań zbioru

Jeden wylosowany zbiór to za mało, żeby mówić o prawidłowości, więc całość powtórzono dla
pięciu różnych ziaren losowania zbioru treningowego:

| ziarno | N < M rozrzut / błąd | N = M rozrzut / błąd | N > M rozrzut / błąd |
|---|---|---|---|
| 2024 | 1.68e-10 / 4.93e-01 | 3.53e-10 / 9.99e-15 | 1.97e+00 / 9.97e-15 |
| 7 | 7.41e-08 / 4.43e-01 | 6.87e-10 / 1.00e-14 | 1.90e+00 / 9.95e-15 |
| 99 | 7.77e-16 / 8.20e-01 | 7.94e-10 / 9.99e-15 | 1.69e+00 / 9.90e-15 |
| 12345 | 3.94e-05 / 6.45e-01 | 8.01e-10 / 1.00e-14 | 1.72e+00 / 9.94e-15 |
| 555 | 1.48e-06 / 9.36e-01 | 6.19e-10 / 9.96e-15 | 1.97e+00 / 9.72e-15 |

Podział jest stabilny: dla *N < M* i *N = M* rozrzut nie przekracza 1e-5, dla *N > M* wynosi
1.7–2.0, czyli tyle, ile same wagi. To **pięć rzędów wielkości** różnicy, więc wniosek nie
zależy od konkretnego losowania. Rozrzut przy *N < M* bywa rzędu 1e-5 zamiast 1e-15, bo
przebiegi zatrzymują się na kryterium stagnacji w nieco różnych momentach — to efekt warunku
stopu, nie niejednoznaczności rozwiązania.

## 4. Odpowiedzi na pytania z zadania

### Czy widać prawidłowości w wagach końcowych?

Tak, i są one skrajnie różne w trzech przypadkach:

- **N < M** — wagi końcowe są **nieodróżnialne** we wszystkich przebiegach (rozrzut 1.68e-10
  przy wagach rzędu 0.5). Punkt startowy nie ma żadnego znaczenia.
- **N = M** — wagi końcowe są **nieodróżnialne** (rozrzut 5.91e-10) i równe dokładnemu
  rozwiązaniu układu.
- **N > M** — wagi końcowe są **różne w każdym przebiegu** (rozrzut 2.28, tego samego rzędu co
  same wagi). Każdy przebieg kończy w innym punkcie, ale wszystkie dają zerowy błąd.

### Czy skuteczny trening był możliwy?

- **N < M — NIE.** Błąd końcowy 0.4929 i nie da się go zmniejszyć: to dokładnie błąd rozwiązania
  najmniejszych kwadratów, czyli matematyczna dolna granica. Neuron nie jest w stanie odtworzyć
  wszystkich 10 wzorców, mając tylko 3 wagi.
- **N = M — TAK.** Błąd spada do 1e-14, neuron odtwarza wszystkie wzorce dokładnie.
- **N > M — TAK.** Błąd spada do 1e-14. Trening jest skuteczny, ale rozwiązanie nie jest jedyne.

## 5. Interpretacja

Wszystko powyższe jest bezpośrednią konsekwencją tego, że trening neuronu liniowego to
rozwiązywanie układu `X w = t` o *M* równaniach i *N* niewiadomych.

**N < M — układ nadokreślony.** Więcej równań niż niewiadomych. Przy losowym `t` układ jest
sprzeczny, więc rozwiązanie dokładne **nie istnieje**. Algorytm minimalizuje błąd kwadratowy,
a ten ma **dokładnie jedno** minimum (bo `X` ma pełny rząd kolumnowy, tu 3). Stąd identyczne wagi
w każdym przebiegu i niezerowy błąd resztkowy. Mówiąc obrazowo: 3 wagi to za mało stopni swobody,
żeby przejść przez 10 narzuconych warunków — neuron może jedynie znaleźć najlepszy kompromis,
a ten kompromis jest jeden.

**N = M — układ kwadratowy.** Tyle równań co niewiadomych. Macierz `X` jest nieosobliwa (rząd 5),
więc istnieje **dokładnie jedno** rozwiązanie. Algorytm do niego zbiega niezależnie od startu —
identyczne wagi i zerowy błąd. To jedyny przypadek, w którym trening jest i skuteczny, i
jednoznaczny.

**N > M — układ niedookreślony.** Mniej równań niż niewiadomych, rząd `X` wynosi 4, więc jądro
`X` ma wymiar 10 − 4 = **6**. Rozwiązań dokładnych jest **nieskończenie wiele** i tworzą one
6-wymiarową podprzestrzeń afiniczną. Algorytm zbiega do tego z nich, który jest najbliżej punktu
startowego — bo, jak pokazano w sekcji 1, nie potrafi ruszyć składowej leżącej w jądrze `X`.
Stąd: zerowy błąd w każdym przebiegu, ale za każdym razem inne wagi. Co ważne, **różnice nie są
przypadkowe** — leżą dokładnie w jądrze `X` (potwierdzone numerycznie: 2.58e-07), czyli w
kierunkach obojętnych dla zbioru treningowego.

**Wniosek ogólny.** Jednoznaczność wag końcowych i skuteczność treningu to dwie **niezależne**
własności, a decyduje o nich relacja między *N* a *M*:

| | rozwiązanie dokładne istnieje | wagi jednoznaczne |
|---|---|---|
| N < M | nie | tak |
| N = M | tak | tak |
| N > M | tak | **nie** |

### Doprecyzowanie: decyduje rząd macierzy X, nie samo porównanie N z M

Powyższy podział jest poprawny, ale warto zobaczyć, co jest jego prawdziwą przyczyną.
Liczba wymiarów swobody, jakie zostają neuronowi po treningu, to wymiar jądra `X`, czyli
`N − rank(X)`. Relacja *N* do *M* rządzi wynikiem tylko dlatego, że dla losowych macierzy
`rank(X) = min(N, M)` zachodzi prawie na pewno.

Sprawdzenie: macierz `X` o wymiarze 5 × 5 (a więc *N = M*), ale o liniowo zależnych wierszach,
przez co `rank(X) = 3` i jądro ma wymiar 2:

| wielkość | wartość |
|---|---|
| rozrzut wag między przebiegami | **1.223e+00** |
| błąd końcowy | 9.903e-15 |
| `‖X(wᵢ − w₁)‖` | 3.047e-07 |

Mimo *N = M* wagi końcowe są różne w każdym przebiegu, a błąd zerowy — czyli dokładnie
zachowanie układu niedookreślonego. Poprawne sformułowanie prawidłowości brzmi więc:

- `rank(X) = N` → rozwiązanie jednoznaczne, wagi końcowe niezależne od startu,
- `rank(X) < N` → rozwiązań nieskończenie wiele, wagi zależą od startu,
- `rank(X) < M` → układ na ogół sprzeczny, błąd nie spada do zera.

Dla losowych zbiorów te trzy warunki sprowadzają się do porównania *N* z *M*, które bada zadanie.

Przypadek *N > M* jest szczególnie wart uwagi z praktycznego punktu widzenia: zerowy błąd na
zbiorze treningowym **nie oznacza**, że model jest wyznaczony — pozostaje 6 wymiarów swobody,
o których dane nie mówią nic. To ta sama sytuacja, co przeuczenie modelu o nadmiarowej liczbie
parametrów.

## 6. Obserwacja dodatkowa — obciążenie reguły przyrostowej

Reguła delty poprawia wagi po każdym wzorcu z osobna, a nie raz na epokę. Przy stałym η nie
zatrzymuje się więc dokładnie w punkcie najmniejszych kwadratów, lecz w jego otoczeniu o
promieniu rzędu η. Pomiar dla przypadku *N < M*:

| η | odległość od `X⁺t` | iloraz | błąd MSE |
|---|---|---|---|
| 0.014662 | 5.415e-03 | — | 0.492906 |
| 0.007331 | 2.668e-03 | 2.03× | 0.492872 |
| 0.003665 | 1.325e-03 | 2.01× | 0.492864 |
| 0.001833 | 6.599e-04 | 2.01× | 0.492862 |

Dwukrotne zmniejszenie η dwukrotnie zmniejsza odchyłkę, a błąd MSE dąży do granicy
najmniejszych kwadratów 0.492861. Potwierdza to, że obciążenie jest rzędu O(η), a nie błędem
implementacji. Nie zmienia to żadnego z wniosków z sekcji 5 — rozrzut wag między przebiegami
(1.68e-10) jest o siedem rzędów wielkości mniejszy niż to obciążenie.

## 7. Uruchomienie

```bash
python3 experiments.py                 # pełne eksperymenty, 8 przebiegów
python3 experiments.py --runs 20       # więcej przebiegów
python3 experiments.py --data-seed 7   # inny losowy zbiór treningowy
```

Wyniki trafiają do katalogu `wyniki/`. Wymagany `numpy`; `matplotlib` tylko do wykresów —
bez niego liczby i tak się policzą.
