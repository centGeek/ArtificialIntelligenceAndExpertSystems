# Sieć MADALINE do rozpoznawania znaków — sprawozdanie

**Inteligentna Analiza Danych**, Instytut Informatyki Politechniki Łódzkiej

## 1. Implementacja

Dwa skrypty wg specyfikacji z punktu 1.5 materiałów do Wykładu nr 2:

```
font_generator.py w h x y font_file letter noise_level output_directory
madaline_ocr.py train_directory test_directory
```

`font_generator.py` rysuje literę wybraną czcionką TrueType na białym tle, binaryzuje obraz
progiem 128, zaszumia go (każdy piksel odwracany z prawdopodobieństwem `noise_level/100`)
i zapisuje jako `<litera>.png` wraz z linią w `description.txt`.

`madaline_ocr.py` buduje sieć ze wzorców w `train_directory` i rozpoznaje znaki z `test_directory`.

Sieć zbudowana jest na istniejącej klasie `Network` z `zadanie2/NeuronNetwork.py`: `Network([N, k])`
bez biasu, z aktywacją liniową, to dokładnie warstwa MADALINE — neuron liniowy liczy iloczyn
skalarny swojego wektora wagowego z wejściem. Wektor wagowy ustawiany jest **wprost** na
znormalizowany wzorzec, bez iteracyjnego treningu. Wyjście neuronu
to iloczyn skalarny `w · x`; przy obu wektorach o długości 1 jest to cosinus kąta między nimi,
czyli współczynnik pewności z przedziału [0, 1].

Nowy kod to tylko obsługa obrazów i plików opisu — sama sieć pochodzi z istniejącej implementacji.

Zaimplementowano z użyciem PIL zamiast OpenCV — wczytanie, konwersja do skali szarości
i progowanie są dostępne w obu bibliotekach, a PIL oszczędza jedną zależność.

**Konwencja pikseli: tusz = 1, tło = 0.** Jest to istotne merytorycznie. Odwrotna konwencja
dałaby wszystkie podobieństwa bliskie jedności, bo tło zajmuje ponad 90% obrazu i zdominowałoby
iloczyn skalarny.

## 2. Weryfikacja poprawności

**Przykład z wykładu.** Dla ręcznych kształtów 4×4 liter X, Y, Z i zaburzonej litery X
implementacja daje wartości identyczne z podanymi w materiałach (Wykład 1, Rozdz. 4.4):

| | materiały | implementacja |
|---|---|---|
| zaburzona X → litera X | 0.943 | **0.943** |
| zaburzona X → litera Y | 0.816 | **0.816** |
| zaburzona X → litera Z | 0.738 | **0.738** |

**Test poprawności dla szumu 0%** (punkt 3 specyfikacji) — wzorce testowane na samych sobie:

```
letter a, noise level 0% -> letter a, noise level 0%, confidence: 1.000
letter b, noise level 0% -> letter b, noise level 0%, confidence: 1.000
letter c, noise level 0% -> letter c, noise level 0%, confidence: 1.000
```

Pewność równa dokładnie 1.000 potwierdza poprawną normalizację wektorów.

## 3. Wyniki dla zaszumionych znaków

Wzorce: litery a, b, c, czcionka Times New Roman, obrazy 128×128, pozycja (28, 10).

| poziom szumu | a → | b → | c → | poprawnie |
|---|---|---|---|---|
| 0% | a (1.000) | b (1.000) | c (1.000) | **3/3** |
| 10% | a (0.522) | b (0.577) | c (0.489) | **3/3** |
| 30% | a (0.264) | b (0.303) | c (0.244) | **3/3** |
| 50% | **b** (0.170) | b (0.174) | **b** (0.167) | 1/3 |
| 90% | **b** (0.170) | **a** (0.128) | **b** (0.153) | 0/3 |

Pełne raporty w katalogu `wyniki/`.

Rozpoznawanie działa bezbłędnie do 30% zaszumienia, przy 50% się załamuje, przy 90% jest
całkowicie przypadkowe.

## 4. Analiza

### Dlaczego pewność spada tak szybko

Szum odwraca **każdy** piksel obrazu, również tło. Litera zajmuje tylko niewielką jego część:

| wzorzec | piksele tuszu | udział w obrazie |
|---|---|---|
| a | 761 | 4.6% |
| b | 999 | 6.1% |
| c | 609 | 3.7% |

Przy 10% szumu do obrazu trafia około 1560 losowych czarnych pikseli z tła — **dwa razy więcej,
niż liczy sama litera**. Stąd pewność spada do ~0.5 już przy niewielkim zaszumieniu.

Zależność daje się wyprowadzić. Jeśli wzorzec ma `n` pikseli tuszu, obraz `P` pikseli ogółem,
a szum wynosi `p`, to oczekiwany cosinus wynosi:

```
cos = (1−p)·n / ( √n · √((1−p)·n + p·(P−n)) )
```

Dla `n = 761`, `P = 16384`, `p = 0.1` wzór daje **0.524**, a pomiar **0.522**. Zgodność
potwierdza, że spadek pewności jest własnością metody, a nie błędem implementacji.

### Dlaczego przy dużym szumie wygrywa zawsze „b"

Przy 50% i 90% szumu niemal wszystkie znaki są rozpoznawane jako „b" — litera o **największej
liczbie pikseli tuszu** (999). Nie jest to przypadek. Dla niemal losowego wektora wejściowego
wartość oczekiwana iloczynu skalarnego z neuronem `i` rośnie jak `√nᵢ`, więc wygrywa neuron
o najliczniejszym wzorcu, niezależnie od zawartości obrazu.

Sprawdzenie: na **300 obrazach czystego szumu losowego** neuron „b" wygrał **300 razy na 300**.

Sieć MADALINE nie ma więc żadnego mechanizmu odrzucenia — zawsze wskazuje jakiegoś zwycięzcę,
nawet gdy wejście nie zawiera żadnego znaku. Jedyną wskazówką jest niska wartość współczynnika
pewności (0.13–0.18 wobec 1.000 dla wzorca czystego).

## 5. Wnioski

1. Metoda działa poprawnie: wzorce czyste rozpoznawane są z pewnością dokładnie 1.000,
   a wyniki dla kształtów 4×4 zgadzają się co do trzeciego miejsca z materiałami.
2. Odporność na szum sięga **30%** — do tego poziomu wszystkie znaki rozpoznawane są prawidłowo.
3. Współczynnik pewności nie jest prawdopodobieństwem, lecz cosinusem kąta między wektorem
   wejściowym a wagowym. Jego spadek do ~0.5 przy zaledwie 10% szumu wynika z tego, że szum
   dotyka całego obrazu, a litera zajmuje kilka procent jego powierzchni.
4. Skuteczność metody zależy więc nie tylko od poziomu szumu, ale też od **proporcji tuszu do
   tła**. Ten sam poziom zaszumienia zaszkodzi bardziej cienkiej literze na dużym tle niż
   grubej wypełniającej kadr.
5. Przy całkowitym zaszumieniu sieć degeneruje się do stałego wskazywania wzorca o największej
   powierzchni. Praktyczny system OCR wymagałby progu odrzucenia opartego na współczynniku
   pewności.

## 6. Uruchomienie

```bash
# wzorce
for L in a b c; do
  python font_generator.py 128 128 28 10 "Times New Roman.ttf" $L 0 training_patterns
done
# znaki testowe o zadanym zaszumieniu
for L in a b c; do
  python font_generator.py 128 128 28 10 "Times New Roman.ttf" $L 30 test_patterns_30
done
# rozpoznawanie
python madaline_ocr.py training_patterns test_patterns_30
```

Wymagane `numpy` i `pillow`.
