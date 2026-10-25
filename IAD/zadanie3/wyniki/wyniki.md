# Wyniki eksperymentow

Parametry: krok uczacy 0.5, 20000 epok, 10 przebiegow.

### Blad koncowy po treningu

| wariant | min | mediana | max | poprawnie sklasyfikowanych |
|---|---|---|---|---|
| z BIAS | 0.000116 | 0.000121 | 0.000122 | 4/4 |
| bez BIAS | 0.384495 | 0.417581 | 0.444623 | 4/4 |

### Wyjscia sieci z BIAS (najlepszy przebieg)

| wzorzec | y1 | y2 | y3 | y4 | najwieksze |
|---|---|---|---|---|---|
| 1 | 0.998 | 0.005 | 0.000 | 0.007 | 1 |
| 2 | 0.005 | 0.998 | 0.007 | 0.000 | 2 |
| 3 | 0.000 | 0.003 | 0.997 | 0.005 | 3 |
| 4 | 0.003 | 0.000 | 0.004 | 0.996 | 4 |

### Polozenia wzorcow w warstwie ukrytej z BIAS

| wzorzec | h1 | h2 |
|---|---|---|
| 1 | -2.075 | -0.535 |
| 2 | 0.800 | 1.251 |
| 3 | 4.707 | -1.591 |
| 4 | -0.713 | -5.492 |

- prosta dowolna (neuron z biasem): **4 z 4**
- prosta przez poczatek (neuron bez biasu): **2 z 4**

### Wyjscia sieci bez BIAS (najlepszy przebieg)

| wzorzec | y1 | y2 | y3 | y4 | najwieksze |
|---|---|---|---|---|---|
| 1 | 1.000 | 0.166 | 0.154 | 0.582 | 1 |
| 2 | 0.000 | 1.000 | 0.018 | 0.016 | 2 |
| 3 | 0.019 | 0.151 | 1.000 | 0.000 | 3 |
| 4 | 0.052 | 0.570 | 0.000 | 0.834 | 4 |

### Polozenia wzorcow w warstwie ukrytej bez BIAS

| wzorzec | h1 | h2 |
|---|---|---|
| 1 | -0.834 | -0.820 |
| 2 | 16.964 | 14.435 |
| 3 | -14.444 | 16.973 |
| 4 | 0.865 | -0.806 |

- prosta dowolna (neuron z biasem): **4 z 4**
- prosta przez poczatek (neuron bez biasu): **2 z 4**

### Czy blad bez biasu to granica, czy wolna zbieznosc?

| krok uczacy | epoki | najlepszy blad |
|---|---|---|
| 0.1 | 20000 | 0.420717 |
| 0.1 | 50000 | 0.381481 |
| 0.5 | 20000 | 0.384495 |
| 0.5 | 50000 | 0.396724 |
| 2.0 | 20000 | 0.634560 |
| 2.0 | 50000 | 0.631445 |
