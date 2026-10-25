# Wyniki eksperymentow

### Weryfikacja wobec przykladu z wykladu (ksztalty 4 x 4)

| wzorzec oryginalny | materialy | implementacja |
|---|---|---|
| litera X | 0.943 | **0.943** |
| litera Y | 0.816 | **0.816** |
| litera Z | 0.738 | **0.738** |

### Udzial tuszu w obrazie

| wzorzec | piksele tuszu | wszystkie piksele | udzial |
|---|---|---|---|
| letter a | 761 | 16384 | 4.6% |
| letter b | 999 | 16384 | 6.1% |
| letter c | 609 | 16384 | 3.7% |

### Rozpoznawanie przy rosnacym zaszumieniu

| poziom szumu | a -> | b -> | c -> | poprawnie |
|---|---|---|---|---|
| 0% | a (1.000) | b (1.000) | c (1.000) | **3/3** |
| 10% | a (0.522) | b (0.577) | c (0.489) | **3/3** |
| 30% | a (0.264) | b (0.303) | c (0.244) | **3/3** |
| 50% | **b** (0.170) | b (0.174) | **b** (0.167) | **1/3** |
| 90% | **b** (0.170) | **a** (0.128) | **b** (0.153) | **0/3** |

### Pomiar kontra model analityczny

| wzorzec | poziom szumu | wyjscie wlasnego neuronu | model |
|---|---|---|---|
| letter a | 0% | 1.000 | 1.000 |
| letter a | 10% | 0.522 | 0.524 |
| letter a | 30% | 0.264 | 0.267 |
| letter a | 50% | 0.138 | 0.152 |
| letter a | 90% | 0.026 | 0.023 |
| letter b | 0% | 1.000 | 1.000 |
| letter b | 10% | 0.577 | 0.576 |
| letter b | 30% | 0.303 | 0.303 |
| letter b | 50% | 0.174 | 0.175 |
| letter b | 90% | 0.027 | 0.027 |
| letter c | 0% | 1.000 | 1.000 |
| letter c | 10% | 0.489 | 0.482 |
| letter c | 30% | 0.244 | 0.241 |
| letter c | 50% | 0.130 | 0.136 |
| letter c | 90% | 0.016 | 0.021 |
