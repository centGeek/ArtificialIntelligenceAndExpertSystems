# Wyniki eksperymentow

Liczba przebiegow na przypadek: 8. Ziarno zbioru treningowego: 2024.

### Przypadek N < M  (N = 3 wag, M = 10 wzorcow)

uklad nadokreslony - wiecej rownan niz niewiadomych. Rzad macierzy X: 3 (X ma wymiar 10 x 3). Wspolczynnik uczenia: 0.014662.

Wagi koncowe w kolejnych przebiegach (kazdy start z innych wag losowych):

| przebieg | w1 | w2 | w3 | blad MSE | epoki |
|---|---|---|---|---|---|
| 1 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 442 |
| 2 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 272 |
| 3 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 414 |
| 4 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 400 |
| 5 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 441 |
| 6 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 404 |
| 7 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 433 |
| 8 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 | 415 |

- Rozrzut wag miedzy przebiegami (max-min na wage): **1.676e-10**
- Blad koncowy: od 4.929e-01 do 4.929e-01
- Blad rozwiazania najmniejszych kwadratow (dolna granica): 0.492861
- Odleglosc wag koncowych od pseudoodwrotnosci X^+ t: 5.415e-03
- Odchylka od przewidywanej granicy w0 + X^+(t - X w0): 5.415e-03
- ||X (w_i - w_1)||, czyli czy roznice leza w jadrze X: 2.915e-10
- Powod zatrzymania treningu: blad przestal spadac przez 200 epok
- Skuteczny trening (neuron odtwarza wszystkie wzorce): **NIE**

### Przypadek N = M  (N = 5 wag, M = 5 wzorcow)

uklad kwadratowy - tyle samo rownan co niewiadomych. Rzad macierzy X: 5 (X ma wymiar 5 x 5). Wspolczynnik uczenia: 0.018192.

Wagi koncowe w kolejnych przebiegach (kazdy start z innych wag losowych):

| przebieg | w1 | w2 | w3 | w4 | w5 | blad MSE | epoki |
|---|---|---|---|---|---|---|---|
| 1 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.982e-15 | 13318 |
| 2 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.980e-15 | 13662 |
| 3 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.988e-15 | 14046 |
| 4 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.981e-15 | 13741 |
| 5 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.991e-15 | 13529 |
| 6 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.987e-15 | 14010 |
| 7 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.998e-15 | 14067 |
| 8 | -1.2187 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 9.992e-15 | 13797 |

- Rozrzut wag miedzy przebiegami (max-min na wage): **5.909e-10**
- Blad koncowy: od 9.980e-15 do 9.998e-15
- Blad rozwiazania najmniejszych kwadratow (dolna granica): 0.000000
- Odleglosc wag koncowych od pseudoodwrotnosci X^+ t: 6.670e-07
- Odchylka od przewidywanej granicy w0 + X^+(t - X w0): 6.670e-07
- ||X (w_i - w_1)||, czyli czy roznice leza w jadrze X: 1.237e-10
- Powod zatrzymania treningu: osiagnieto zadany blad
- Skuteczny trening (neuron odtwarza wszystkie wzorce): **TAK**

### Przypadek N > M  (N = 10 wag, M = 4 wzorcow)

uklad niedookreslony - mniej rownan niz niewiadomych. Rzad macierzy X: 4 (X ma wymiar 4 x 10). Wspolczynnik uczenia: 0.010223.

Wagi koncowe w kolejnych przebiegach (kazdy start z innych wag losowych):

| przebieg | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | blad MSE | epoki |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.1484 | 0.2321 | 0.8647 | -0.0028 | -0.1200 | 0.1589 | 0.4281 | -0.5156 | -0.3027 | -0.4232 | 9.966e-15 | 1204 |
| 2 | -0.9315 | 0.6882 | -0.4528 | 0.3341 | 0.5786 | -0.8529 | -0.3146 | -0.4406 | 0.7765 | 0.3169 | 9.961e-15 | 1210 |
| 3 | -0.3221 | 0.2628 | 0.2131 | -0.1244 | 0.8318 | -0.1629 | 0.4370 | -1.3320 | 0.3120 | 0.0711 | 9.951e-15 | 1332 |
| 4 | -0.9315 | 0.3100 | 0.3404 | 0.4895 | 0.6524 | 0.2460 | 0.6681 | 0.6353 | 0.3371 | -0.6252 | 9.887e-15 | 1344 |
| 5 | -0.9948 | 0.1395 | 0.7349 | 0.0409 | 0.4911 | -0.2731 | -0.3504 | 0.4790 | 0.6970 | -1.1956 | 9.858e-15 | 929 |
| 6 | -0.4440 | 0.3526 | 0.1028 | 0.1796 | 1.0787 | -0.0493 | 0.2630 | -0.7475 | 0.4669 | 0.4817 | 9.965e-15 | 1271 |
| 7 | -0.3588 | 0.1756 | 0.8474 | 0.1737 | 1.3561 | 0.2222 | -0.1955 | -0.8086 | 0.2946 | 1.0806 | 9.841e-15 | 1401 |
| 8 | -1.2887 | -0.1010 | 0.5066 | -0.5214 | 0.5686 | 1.0695 | -0.3734 | 0.4197 | 0.5905 | -0.3002 | 9.855e-15 | 1385 |

- Rozrzut wag miedzy przebiegami (max-min na wage): **2.276e+00**
- Blad koncowy: od 9.841e-15 do 9.966e-15
- Blad rozwiazania najmniejszych kwadratow (dolna granica): 0.000000
- Odleglosc wag koncowych od pseudoodwrotnosci X^+ t: 1.219e+00
- Odchylka od przewidywanej granicy w0 + X^+(t - X w0): 1.354e-07
- ||X (w_i - w_1)||, czyli czy roznice leza w jadrze X: 2.580e-07
- Powod zatrzymania treningu: osiagnieto zadany blad
- Skuteczny trening (neuron odtwarza wszystkie wzorce): **TAK**

### Wplyw wspolczynnika uczenia na dokladnosc (przypadek N < M)

| eta | odleglosc od X^+ t | blad MSE |
|---|---|---|
| 0.014662 | 5.415e-03 | 0.492906 |
| 0.0073308 | 2.668e-03 | 0.492872 |
| 0.0036654 | 1.325e-03 | 0.492864 |
| 0.0018327 | 6.599e-04 | 0.492862 |

### Powtarzalnosc dla roznych losowan zbioru treningowego

| ziarno | N < M rozrzut / blad | N = M rozrzut / blad | N > M rozrzut / blad |
|---|---|---|---|
| 2024 | 1.68e-10 / 4.93e-01 | 3.53e-10 / 9.99e-15 | 1.97e+00 / 9.97e-15 |
| 7 | 7.41e-08 / 4.43e-01 | 6.87e-10 / 1.00e-14 | 1.90e+00 / 9.95e-15 |
| 99 | 7.77e-16 / 8.20e-01 | 7.94e-10 / 9.99e-15 | 1.69e+00 / 9.90e-15 |
| 12345 | 3.94e-05 / 6.45e-01 | 8.01e-10 / 1.00e-14 | 1.72e+00 / 9.94e-15 |
| 555 | 1.48e-06 / 9.36e-01 | 6.19e-10 / 9.96e-15 | 1.97e+00 / 9.72e-15 |

### Co naprawde rzadzi wynikiem: N vs M czy rzad macierzy X?

Macierz X o wymiarze 5 x 5 (czyli N = M), ale o liniowo zaleznych wierszach: rzad = 3, wymiar jadra = 2.

- rozrzut wag miedzy przebiegami: **1.223e+00**
- blad koncowy: 9.903e-15
- ||X (w_i - w_1)||: 3.047e-07

Mimo N = M wagi koncowe sa rozne w kazdym przebiegu, a blad zerowy - czyli zachowanie przypadku niedookreslonego. Decyduje **rzad macierzy X**, a nie samo porownanie N z M. Relacja N vs M jest tylko typowym nastepstwem rzedu, bo losowe macierze sa prawie na pewno pelnego rzedu.

