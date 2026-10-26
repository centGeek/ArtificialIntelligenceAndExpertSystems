# Perceptron 4-2-4 i wsteczna propagacja błędu — sprawozdanie

**Inteligentna Analiza Danych**, Instytut Informatyki Politechniki Łódzkiej

## 1. Zadanie i implementacja

Sieć o architekturze z punktu 2.2 materiałów do Wykładu nr 2:

- **4 neurony kopiujące** w warstwie wejściowej (przepisują wejścia dalej, nie podlegają treningowi),
- **2 neurony liniowe** w warstwie ukrytej — bez funkcji aktywacji i bez biasu, `hᵢ = Σⱼ wᵢⱼ⁽¹⁾xⱼ`,
- **4 neurony sigmoidalne** w warstwie wyjściowej, z wejściem obciążającym (BIAS), `yᵢ = f(sᵢ)`.

Liczba wag: 2·4 = 8 w warstwie ukrytej i 4·3 = 12 w wyjściowej, razem **20** — zgodnie z zapisem
`w ∈ ℝ²⁰` w materiałach.

Zbiór treningowy to 4 wzorce autoasocjacji: `x = z = [1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[0,0,0,1]`.

Trening realizuje **Algorytm 2.1**, kroki 1–10. Dwie rzeczy warte podkreślenia:

1. **W kroku 8 nie ma czynnika `f'(·)`** — warstwa ukryta jest liniowa, więc jej pochodna wynosi 1
   i błąd propaguje się wstecz czystym `δᵢ⁽¹⁾ = Σₖ wₖᵢ⁽²⁾δₖ⁽²⁾`. Kolumna biasu nie prowadzi do
   żadnego neuronu ukrytego, więc nic przez nią nie wraca.
2. **Jedynym kryterium stopu jest liczba epok `L`** (krok 2). Nie ma progu błędu.

Algorytm 2.1 zapisany jest dla pojedynczego wzorca, a zadanie wymaga czterech. Materiały
stwierdzają, że rozszerzenie jest bezpośrednie, więc w każdej epoce prezentowane są kolejno
wszystkie wzorce, a wagi poprawiane po każdym z nich — tak samo jak w Algorytmie 3.2.

Sieć zbudowana jest na istniejącej klasie `Network` z `zadanie2/NeuronNetwork.py`:
`Network([4, 2, 4])` z aktywacją i biasem zadanymi osobno dla każdej warstwy —
`[(liniowa, pochodna), (sigmoidalna, pochodna)]` oraz `bias=[False, True]`. Przy tej
konfiguracji `forward()`, `back_propagation()`, `update_weights()` i `update_biases()`
realizują dokładnie Algorytm 2.1. Sprawdzone numerycznie wobec wzorów 2.1.1–2.1.6b
rozpisanych wprost: wagi warstwy ukrytej, wyjściowej i biasy zgadzają się **co do bitu**
(różnica 0.00e+00 w każdym z trzech zestawów). Nowy kod to tylko pętla epok.

Poprawność gradientów sprawdzono wobec różnic skończonych: maksymalna rozbieżność **3·10⁻¹¹**
(w obu wariantach, z biasem i bez).

Parametry: `η = 0.5`, `L = 20000`, 10 przebiegów od różnych wag losowych z `[−1, 1]`.

## 2. Wyniki

| wariant | błąd E — min | mediana | max |
|---|---|---|---|
| warstwa wyjściowa **z BIAS** | **0.000116** | 0.000121 | 0.000122 |
| warstwa wyjściowa **bez BIAS** | **0.384495** | 0.417581 | 0.444623 |

Błąd bez biasu jest **ponad 3300 razy większy**.

**Wyjścia sieci z BIAS** (najlepszy przebieg) — sieć odtwarza wszystkie cztery wzorce:

| wzorzec | y₁ | y₂ | y₃ | y₄ |
|---|---|---|---|---|
| 1 | **0.998** | 0.005 | 0.000 | 0.007 |
| 2 | 0.005 | **0.998** | 0.007 | 0.000 |
| 3 | 0.000 | 0.003 | **0.997** | 0.005 |
| 4 | 0.003 | 0.000 | 0.004 | **0.996** |

**Wyjścia sieci bez BIAS** — poprawne są dokładnie **dwa** wzorce:

| wzorzec | y₁ | y₂ | y₃ | y₄ | |
|---|---|---|---|---|---|
| 1 | **1.000** | 0.166 | 0.154 | 0.582 | źle — fałszywe y₄ |
| 2 | 0.000 | **1.000** | 0.018 | 0.016 | dobrze |
| 3 | 0.019 | 0.151 | **1.000** | 0.000 | dobrze |
| 4 | 0.052 | 0.570 | 0.000 | **0.834** | źle — fałszywe y₂ |

Wzorce 2 i 3 są odtwarzane czysto. Dla wzorców 1 i 4 sieć wprawdzie zapala właściwe wyjście,
ale **nie potrafi wygasić fałszywego** — y₄ = 0.582 przy wzorcu 1 i y₂ = 0.570 przy wzorcu 4,
choć oba powinny wynosić 0. To stąd bierze się cały błąd resztkowy.

### Czy to granica, czy wolna zbieżność?

Sprawdziliśmy, czy wysoki błąd nie wynika ze zbyt krótkiego treningu albo ze źle dobranego kroku
uczącego. Trening bez biasów powtórzyliśmy dla trzech wartości kroku uczącego (0.1, 0.5 i 2.0)
i dla dwóch liczb epok (20000 i 50000), czyli dla sześciu kombinacji. Każdą uruchomiliśmy 5 razy
od różnych wag startowych i zapisaliśmy najmniejszy uzyskany błąd:

| krok uczący | epoki | najlepszy błąd |
|---|---|---|
| 0.1 | 20000 | 0.420717 |
| 0.1 | 50000 | 0.381481 |
| 0.5 | 20000 | 0.384495 |
| 0.5 | 50000 | 0.396724 |
| 2.0 | 20000 | 0.634560 |
| 2.0 | 50000 | 0.631445 |

Ani zwiększenie liczby epok, ani zmiana kroku uczącego nie zbijają błędu poniżej ~0.37.
To **faktyczna granica**, a nie niedouczenie.

## 3. Dlaczego — geometria warstwy ukrytej

Warstwa ukryta ma 2 neurony, więc każdy wzorzec staje się **punktem na płaszczyźnie**.
Neuron wyjściowy `i` musi dać wartość bliską 1 dla „swojego" punktu i bliską 0 dla pozostałych
trzech, czyli jego granica decyzyjna musi **oddzielić jeden punkt od trzech**.

Położenia wzorców w warstwie ukrytej (sieć z biasem):

| wzorzec | h₁ | h₂ |
|---|---|---|
| 1 | −2.075 | −0.535 |
| 2 | 0.800 | 1.251 |
| 3 | 4.707 | −1.591 |
| 4 | −0.713 | −5.492 |

Sprawdzenie, ile punktów da się odciąć od pozostałych:

| rodzaj prostej | odpowiada neuronowi | wynik |
|---|---|---|
| dowolna: `u·h + b = 0` | **z BIAS** | **4 z 4** |
| przez początek układu: `u·h = 0` | **bez BIAS** | **2 z 4** |

Ten sam wynik — 4 z 4 i 2 z 4 — wychodzi niezależnie od przebiegu treningu, bo jest własnością
czterech punktów na płaszczyźnie, a nie konkretnych wag.

**I to dokładnie zgadza się z pomiarem**: bez biasu poprawne są 2 wzorce, bo tylko 2 neurony
wyjściowe są w stanie wykonać swoje zadanie.

## 4. Wnioski końcowe

### Czy perceptron dałoby się skutecznie wytrenować bez BIAS w warstwie wyjściowej?

**Nie.** Błąd zatrzymuje się na ~0.38 wobec 0.00012 przy sieci z biasem — ponad 3300 razy
wyżej — i nie spada mimo zwiększania liczby epok ani zmian kroku uczącego. Dokładnie dwa
z czterech wzorców są odtwarzane poprawnie. W pozostałych dwóch sieć zapala wprawdzie właściwe
wyjście, ale nie potrafi wygasić fałszywego — zostaje ono na poziomie 0.57–0.58, choć powinno
wynosić 0.

### Jaka jest rola wejść obciążających?

Bias daje granicy decyzyjnej **swobodę położenia**. Bez niego suma ważona to `sᵢ = u·h`,
więc granica `u·h = 0` zawsze przechodzi przez **początek układu współrzędnych**. Z biasem jest
to `u·h + b = 0` — prosta o tym samym nachyleniu, ale przesunięta tam, gdzie trzeba.

Najprostsza analogia: `u·h` to prosta `y = ax`, a `u·h + b` to `y = ax + b`. Prostą przechodzącą
przez zero nie da się dopasować do danych, które wymagają wyrazu wolnego — i nie pomoże żadne
dobieranie nachylenia.

W tym konkretnym zadaniu konsekwencja jest policzalna. Cztery punkty na płaszczyźnie tworzą
czworokąt wypukły, więc **każdy** z nich da się odciąć jakąś prostą — ale prostych przechodzących
przez początek układu wystarczy tylko dla **dwóch** z nich (dla pozostałych prosta musiałaby
przejść „pomiędzy", a przez zero tego nie zrobi). Stąd 2 poprawne wzorce zamiast 4.

Drugi, bardziej obrazowy sposób: bez biasu zerowe wejście daje zerową sumę ważoną, a więc
`f(0) = 0.5` na każdym wyjściu — sieć nie ma żadnego sposobu, by mieć jakiekolwiek „domyślne
nastawienie". Bias jest właśnie tym nastawieniem: mówi neuronowi, jak bardzo ma być skłonny
się wzbudzić, zanim jeszcze zobaczy jakiekolwiek wejście.

## 5. Uruchomienie

```bash
python3 experiments.py
```

Wymagany `numpy`.
