# Wyniki eksperymentow

### Przypadek N < M  (N = 3 wag, M = 10 wzorcow)

uklad nadokreslony - wiecej rownan niz niewiadomych. Rzad macierzy X: 3. Wspolczynnik uczenia: 0.014662.

| przebieg | w1 | w2 | w3 | blad MSE |
|---|---|---|---|---|
| 1 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 2 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 3 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 4 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 5 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 6 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 7 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |
| 8 | 0.4571 | -0.1883 | -0.3252 | 4.929e-01 |

- rozrzut wag miedzy przebiegami: **4.718e-16**
- blad koncowy: 4.929e-01, granica najmniejszych kwadratow: 0.492861
- odleglosc od X^+ t: 5.415e-03
- odchylka od w0 + X^+(t - X w0): 5.415e-03
- ||X (w_i - w_1)||: 1.117e-15
- skuteczny trening: **NIE**

### Przypadek N = M  (N = 5 wag, M = 5 wzorcow)

uklad kwadratowy - tyle samo rownan co niewiadomych. Rzad macierzy X: 5. Wspolczynnik uczenia: 0.018192.

| przebieg | w1 | w2 | w3 | w4 | w5 | blad MSE |
|---|---|---|---|---|---|---|
| 1 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 7.038e-21 |
| 2 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 1.459e-20 |
| 3 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 3.296e-20 |
| 4 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 1.725e-20 |
| 5 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 1.102e-20 |
| 6 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 3.053e-20 |
| 7 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 3.449e-20 |
| 8 | -1.2188 | -0.1165 | -0.2026 | 0.6470 | -1.4753 | 1.945e-20 |

- rozrzut wag miedzy przebiegami: **6.793e-10**
- blad koncowy: 3.449e-20, granica najmniejszych kwadratow: 0.000000
- odleglosc od X^+ t: 1.239e-09
- odchylka od w0 + X^+(t - X w0): 1.239e-09
- ||X (w_i - w_1)||: 1.596e-10
- skuteczny trening: **TAK**

### Przypadek N > M  (N = 10 wag, M = 4 wzorcow)

uklad niedookreslony - mniej rownan niz niewiadomych. Rzad macierzy X: 4. Wspolczynnik uczenia: 0.010223.

| przebieg | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 | w9 | w10 | blad MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -1.1484 | 0.2321 | 0.8647 | -0.0028 | -0.1200 | 0.1589 | 0.4281 | -0.5156 | -0.3027 | -0.4232 | 1.304e-30 |
| 2 | -0.9315 | 0.6882 | -0.4528 | 0.3341 | 0.5786 | -0.8529 | -0.3146 | -0.4406 | 0.7765 | 0.3169 | 1.108e-29 |
| 3 | -0.3221 | 0.2628 | 0.2131 | -0.1244 | 0.8318 | -0.1629 | 0.4370 | -1.3320 | 0.3120 | 0.0711 | 1.815e-30 |
| 4 | -0.9315 | 0.3100 | 0.3404 | 0.4895 | 0.6524 | 0.2460 | 0.6681 | 0.6353 | 0.3371 | -0.6252 | 3.301e-30 |
| 5 | -0.9948 | 0.1395 | 0.7349 | 0.0409 | 0.4911 | -0.2731 | -0.3504 | 0.4790 | 0.6970 | -1.1956 | 1.541e-30 |
| 6 | -0.4440 | 0.3526 | 0.1028 | 0.1796 | 1.0787 | -0.0493 | 0.2630 | -0.7475 | 0.4669 | 0.4817 | 1.837e-30 |
| 7 | -0.3588 | 0.1756 | 0.8474 | 0.1737 | 1.3561 | 0.2222 | -0.1955 | -0.8086 | 0.2946 | 1.0806 | 4.500e-30 |
| 8 | -1.2887 | -0.1010 | 0.5066 | -0.5214 | 0.5686 | 1.0695 | -0.3734 | 0.4197 | 0.5905 | -0.3002 | 4.111e-30 |

- rozrzut wag miedzy przebiegami: **2.276e+00**
- blad koncowy: 1.108e-29, granica najmniejszych kwadratow: 0.000000
- odleglosc od X^+ t: 1.219e+00
- odchylka od w0 + X^+(t - X w0): 1.088e-14
- ||X (w_i - w_1)||: 5.773e-15
- skuteczny trening: **TAK**

### Wplyw wspolczynnika uczenia (przypadek N < M)

| eta | odleglosc od X^+ t | blad MSE |
|---|---|---|
| 0.014662 | 5.415e-03 | 0.492906 |
| 0.0073308 | 2.668e-03 | 0.492872 |
| 0.0036654 | 1.325e-03 | 0.492864 |
| 0.0018327 | 6.599e-04 | 0.492862 |

### Powtarzalnosc dla roznych losowan zbioru

| ziarno | N < M rozrzut / blad | N = M rozrzut / blad | N > M rozrzut / blad |
|---|---|---|---|
| 2024 | 4.72e-16 / 4.93e-01 | 6.51e-10 / 3.30e-20 | 1.97e+00 / 1.11e-29 |
| 7 | 4.44e-16 / 4.43e-01 | 3.25e-14 / 6.14e-29 | 1.90e+00 / 1.37e-30 |
| 99 | 0.00e+00 / 8.20e-01 | 8.20e-13 / 2.05e-25 | 1.69e+00 / 2.04e-30 |
| 12345 | 4.33e-15 / 6.45e-01 | 0.00e+00 / 3.23e-27 | 1.72e+00 / 6.29e-30 |
| 555 | 3.89e-16 / 9.36e-01 | 0.00e+00 / 2.80e-28 | 1.97e+00 / 1.09e-30 |

### Decyduje rzad macierzy X, nie porownanie N z M

Macierz 5 x 5 (N = M) o liniowo zaleznych wierszach, rzad 3, jadro wymiaru 2:

- rozrzut wag: **1.223e+00**
- blad koncowy: 1.306e-29
- ||X (w_i - w_1)||: 1.089e-14

Mimo N = M wagi roznia sie miedzy przebiegami, a blad jest zerowy - zachowanie ukladu niedookreslonego.
