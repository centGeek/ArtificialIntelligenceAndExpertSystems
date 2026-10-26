# Trening neuronu liniowego wieloma wzorcami — sprawozdanie

**Inteligentna Analiza Danych**, Instytut Informatyki Politechniki Łódzkiej

---

## 1. Zadanie i implementacja

Przedmiotem zadania jest **neuron liniowy**, czyli neuron bez funkcji aktywacji:

```
y = w · x = w₁x₁ + w₂x₂ + ... + w_N x_N
```

Trening wieloma wzorcami realizuje **Algorytm 3.2** — reguła delty dla wielu wzorców uczących.
Implementacja jest wierną transkrypcją kroków 4–10:

| krok | treść | realizacja |
|---|---|---|
| 4 | dopóki `k < K` | pętla `for k in range(epochs)` |
| 5–6, 9 | `μ = 1`, aż wszystkie *M* wzorców zaprezentowane, `μ = μ+1` | pętla `for mu in range(len(X))` |
| 7 | `y = Σ wᵢ xᵢ^(μ)` | `y = self.forward(X[mu])` |
| 8 | `wᵢ = wᵢ + η (z^(μ) − y) xᵢ^(μ)`, `i = 1..N` | `self.weights += learning_rate * (t[mu] - y) * X[mu]` |
| 10 | `k = k+1` | kolejna iteracja pętli zewnętrznej |

Trzy rzeczy warte podkreślenia, bo łatwo je przeoczyć:

1. **Jedynym kryterium stopu jest liczba epok *K*.** Algorytm nie ma progu błędu — wykonuje
   dokładnie *K* epok, nawet gdy błąd dawno przestał spadać. Dodanie takiego progu byłoby
   odstępstwem od specyfikacji, więc go nie ma. Błąd po każdej epoce jest zapisywany, ale
   służy wyłącznie do oceny wyników i wykresów; na przebieg treningu nie wpływa.
2. **Aktualizacja jest przyrostowa** — krok 8 wykonuje się wewnątrz pętli po wzorcach, czyli
   wagi zmieniają się po każdym wzorcu, a nie raz na epokę.
3. **Wszystkie *N* wag korzysta z tego samego `y`** policzonego w kroku 7, przed jakąkolwiek
   zmianą wag.

Neuron nie ma biasu — wzór (3.2.1) to czysta suma `Σ wᵢxᵢ`, bez wyrazu wolnego. Liczba wag wynosi
dokładnie *N*, dzięki czemu odpowiedniość z układem równań jest ścisła.

Implementacja opiera się na istniejącej klasie `Network` z `zadanie2/NeuronNetwork.py`.
Neuron liniowy to `Network([N, 1])` bez biasu, z aktywacją liniową — przy takiej konfiguracji
`forward()`, `back_propagation()` i `update_weights()` realizują dokładnie regułę delty ze
wzoru 3.2.2. Sprawdzone numerycznie: wagi końcowe zgadzają się **co do bitu** z wzorem
rozpisanym wprost (różnica 0.00e+00). Nowy kod to tylko pętla epok i funkcje analityczne.

Pliki: `linear_neuron.py`, `experiments.py`. Przyjęto `K = 5000`.

### Uwaga do kroku 6

Warunek w kroku 6 brzmi „czy `μ = M`?", a przy odpowiedzi twierdzącej algorytm skacze do kroku 10.
Czytane dosłownie oznacza to, że przy `μ = M` epoka kończy się **przed** przetworzeniem ostatniego
wzorca — w każdej epoce neuron widziałby tylko *M−1* wzorców. Przeczy to wyjaśnieniu podanemu
w tym samym kroku („czy wszystkie wzorce treningowe zostały zaprezentowane"), które przy `μ = M`
jest jeszcze fałszywe.

Że chodzi o *M* wzorców, potwierdza eksperyment — przy odczycie dosłownym cała teza zadania się rozpada:

| przypadek | wszystkie *M* wzorców | dosłownie, *M−1* wzorców |
|---|---|---|
| N = M | błąd **1.25e-20** | błąd **8.51e-02** |
| N > M | błąd **5.81e-30** | błąd **4.07e-02** |

Przy *M−1* wzorcach neuron rozwiązuje o jedno równanie mniej, więc *N = M* przestaje wychodzić jako
trening skuteczny i równoważność z układem równań przestaje się odtwarzać. Przyjęto więc intencję:
w każdej epoce przetwarzane są wszystkie *M* wzorców.

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
- wykonano **8 przebiegów** treningu, każdy przez dokładnie `K = 5000` epok,
- współczynnik uczenia jest stały i wspólny dla wszystkich przypadków: `η = 0.1`,
- `t` losowane jest **niezależnie** od `X`. To istotne: gdyby przyjąć `t = X w_prawdziwe`,
  układ byłby zgodny także dla *M > N* i najciekawszy przypadek by zniknął.

## 3. Wyniki

| przypadek | N | M | rząd X | rozrzut wag między przebiegami | błąd końcowy | granica najmn. kwadratów | skuteczny trening |
|---|---|---|---|---|---|---|---|
| N < M | 3 | 10 | 3 | **5.55e-17** | 4.96e-01 | 0.4929 | **NIE** |
| N = M | 5 | 5 | 5 | **0.00e+00** | 1.01e-29 | 0.0000 | **TAK** |
| N > M | 10 | 4 | 4 | **2.28e+00** | 5.93e-32 | 0.0000 | **TAK** |

Pełne tabele wag z każdego przebiegu: `wyniki/wyniki.md`.
Wykresy: `wyniki/zbieznosc_bledu.png`, `wyniki/wagi_koncowe.png`.

### Weryfikacja numeryczna przewidywań teoretycznych

| sprawdzenie | N < M | N = M | N > M |
|---|---|---|---|
| odchyłka od `w₀ + X⁺(t − X w₀)` | 4.40e-02 | 2.13e-14 | **2.44e-15** |
| `‖X (wᵢ − w₁)‖` — czy różnice leżą w jądrze X | 8.64e-17 | 0.00e+00 | **6.11e-16** |

Druga linijka jest najważniejsza dla przypadku *N > M*: wagi końcowe różnią się między
przebiegami o **2.28**, ale po przemnożeniu tych różnic przez `X` otrzymujemy zero z dokładnością do 6e-16.
Oznacza to, że przebiegi różnią się **wyłącznie** składową leżącą w jądrze `X` — czyli w
kierunkach, których wzorce treningowe w ogóle nie widzą.

### Powtarzalność dla różnych losowań zbioru

Jeden wylosowany zbiór to za mało, żeby mówić o prawidłowości, więc całość powtórzono dla
pięciu różnych ziaren losowania zbioru treningowego:

| ziarno | N < M rozrzut / błąd | N = M rozrzut / błąd | N > M rozrzut / błąd |
|---|---|---|---|
| 2024 | 5.55e-17 / 4.96e-01 | 0.00e+00 / 1.01e-29 | 1.97e+00 / 5.01e-32 |
| 7 | 0.00e+00 / 4.60e-01 | 0.00e+00 / 6.58e-31 | 1.90e+00 / 3.45e-32 |
| 99 | 0.00e+00 / 8.27e-01 | 1.27e-12 / 4.93e-25 | 1.69e+00 / 3.77e-32 |
| 12345 | 4.44e-16 / 7.54e-01 | 0.00e+00 / 3.71e-29 | 1.72e+00 / 8.50e-32 |
| 555 | 8.33e-17 / 9.95e-01 | 3.55e-15 / 4.97e-30 | 1.97e+00 / 2.77e-32 |

Podział jest stabilny: dla *N < M* i *N = M* rozrzut nie przekracza 1.3e-12, czyli siedzi na
poziomie precyzji arytmetyki zmiennoprzecinkowej, a dla *N > M* wynosi 1.7–2.0, czyli tyle, ile
same wagi. To **kilkanaście rzędów wielkości** różnicy — wniosek nie zależy od konkretnego
losowania.

## 4. Odpowiedzi na pytania z zadania

### Czy widać prawidłowości w wagach końcowych?

Tak, i są one skrajnie różne w trzech przypadkach:

- **N < M** — wagi końcowe są **nieodróżnialne** we wszystkich przebiegach (rozrzut 5.55e-17,
  czyli precyzja maszynowa, przy wagach rzędu 0.5). Punkt startowy nie ma żadnego znaczenia.
- **N = M** — wagi końcowe są **nieodróżnialne** (rozrzut 0) i równe dokładnemu
  rozwiązaniu układu.
- **N > M** — wagi końcowe są **różne w każdym przebiegu** (rozrzut 2.28, tego samego rzędu co
  same wagi). Każdy przebieg kończy w innym punkcie, ale wszystkie dają zerowy błąd.

### Czy skuteczny trening był możliwy?

- **N < M — NIE.** Błąd końcowy 0.4962, tuż nad granicą najmniejszych kwadratów 0.4929,
  czyli matematyczną dolną granicą, poniżej której nie da się zejść. Neuron nie jest w stanie odtworzyć
  wszystkich 10 wzorców, mając tylko 3 wagi.
- **N = M — TAK.** Błąd spada do 1.0e-29, neuron odtwarza wszystkie wzorce dokładnie.
- **N > M — TAK.** Błąd spada do 5.9e-32. Trening jest skuteczny, ale rozwiązanie nie jest jedyne.

## 5. Interpretacja

Trening neuronu jest równoważny rozwiązywaniu układu `X w = t` o *M* równaniach i *N* niewiadomych.

- **N < M** — układ jest nadokreślony, ponieważ równań jest więcej niż niewiadomych.
  Rozwiązanie dokładne nie istnieje, więc błąd nie spada do zera.
- **N = M** — układ ma dokładnie jedno rozwiązanie, więc wagi końcowe są zawsze takie same.
- **N > M** — układ jest niedookreślony, ponieważ równań jest mniej niż niewiadomych.
  Rozwiązań jest nieskończenie wiele, więc każdy przebieg kończy się innymi wagami.

### Doprecyzowanie: decyduje rząd macierzy X, nie samo porównanie N z M

Powyższy podział jest poprawny, ale warto zobaczyć, co jest jego prawdziwą przyczyną.
Liczba wymiarów swobody, jakie zostają neuronowi po treningu, to wymiar jądra `X`, czyli
`N − rank(X)`.

Sprawdzenie: macierz `X` o wymiarze 5 × 5 (a więc *N = M*), ale o liniowo zależnych wierszach,
przez co `rank(X) = 3` i jądro ma wymiar 2:

| wielkość | wartość |
|---|---|
| rozrzut wag między przebiegami | **1.223e+00** |
| błąd końcowy | 1.306e-29 |
| `‖X(wᵢ − w₁)‖` | 1.089e-14 |

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
(5.55e-17) jest o kilkanaście rzędów wielkości mniejszy niż to obciążenie.

## 7. Uruchomienie

```bash
python3 experiments.py                 # pełne eksperymenty, 8 przebiegów
python3 experiments.py --runs 20       # więcej przebiegów
python3 experiments.py --epochs 50000  # inna liczba epok K
python3 experiments.py --data-seed 7   # inny losowy zbiór treningowy
```

Wyniki trafiają do katalogu `wyniki/`. Wymagany `numpy`; `matplotlib` tylko do wykresów —
bez niego liczby i tak się policzą.
