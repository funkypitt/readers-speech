
**Mel 1re moitié (réglage)** — référence : 114 § (99 par la voix, 14 par les pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane car. (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules (règle bureau)    | 0.84 0.34 **0.48** | 0.77 0.24 0.37 | 0.67 1.00 0.80 |   46 (114) |  819 (224) |
| 1b. pauses seules, fin de phrase + pause ≥ 600 ms | 0.83 0.44 **0.58** | 0.78 0.36 0.50 | 0.58 1.00 0.74 |   61 (114) |  514 (224) |
| 1b. pauses seules, fin de phrase + pause ≥ 400 ms | 0.71 0.55 **0.62** | 0.66 0.48 0.56 | 0.36 1.00 0.53 |   88 (114) |  263 (224) |
| 1b. pauses seules, fin de phrase + pause ≥ 300 ms | 0.71 0.70 **0.71** | 0.67 0.66 0.66 | 0.30 1.00 0.47 |  112 (114) |  193 (224) |
| 2. pauses + indices textuels (q≥0, marqueur≥500) | 0.91 0.54 **0.68** | 0.89 0.47 0.62 | 0.70 1.00 0.82 |   68 (114) |  440 (224) |
| 2a. pauses + question seule        | 0.94 0.52 **0.67** | 0.92 0.45 0.61 | 0.78 1.00 0.88 |   64 (114) |  456 (224) |
| 2b. pauses + marqueurs seuls (≥300 ms) | 0.80 0.42 **0.55** | 0.74 0.35 0.48 | 0.52 0.93 0.67 |   61 (114) |  514 (224) |
| 3. pauses + MFCC+F0 (pyin) (seuil 7.00, pause≥0, vs para, F0×6) | 0.89 0.98 **0.93** | 0.87 0.98 0.92 | 0.50 1.00 0.67 |  126 (114) |  184 (224) |
| 3. pauses + MFCC seuls (seuil 5.00, pause≥0, vs prev, F0×1) | 0.81 0.81 **0.81** | 0.78 0.78 0.78 | 0.39 1.00 0.56 |  114 (114) |  188 (224) |
| 3. pauses + MFCC+F0+écarts-types (seuil 8.75, pause≥0, vs para, F0×6) | 0.90 0.98 **0.94** | 0.88 0.98 0.93 | 0.52 1.00 0.68 |  125 (114) |  181 (224) |
| 3. pauses + MFCC+F0 (YIN simple) (seuil 6.00, pause≥0, vs para, F0×4) | 0.88 0.96 **0.92** | 0.86 0.95 0.90 | 0.48 1.00 0.65 |  124 (114) |  188 (224) |
| 3. pauses + F0 seul (seuil 1.30, pause≥0) | 0.93 0.90 **0.91** | 0.92 0.89 0.90 | 0.64 1.00 0.78 |  111 (114) |  240 (224) |
| 3+2. acoustique (seuil 6.25, pause≥0, F0×3) + indices textuels | 0.92 0.97 **0.94** | 0.91 0.97 0.94 | 0.58 1.00 0.74 |  121 (114) |  191 (224) |
| 4a. pauses + WeSpeaker ResNet34 (cos > 0.60, pause≥0, vs prev) | 1.00 0.98 **0.99** | 1.00 0.98 0.99 | 1.00 1.00 1.00 |  112 (114) |  233 (224) |
| 4b. pauses + segmentation locale PyanNet (changement > 0.3, pause≥0) | 0.97 0.98 **0.97** | 0.96 0.98 0.97 | 0.78 1.00 0.88 |  116 (114) |  203 (224) |
| 4b+2. segmentation locale + indices textuels | 0.93 0.98 **0.95** | 0.92 0.98 0.95 | 0.61 1.00 0.76 |  121 (114) |  191 (224) |

**Mel 2e moitié (test)** — référence : 113 § (103 par la voix, 9 par les pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane car. (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules (règle bureau)    | 0.85 0.29 **0.44** | 0.80 0.23 0.36 | 0.60 1.00 0.75 |   40 (113) |  957 (163) |
| 1b. pauses seules, fin de phrase + pause ≥ 600 ms | 0.80 0.36 **0.49** | 0.77 0.32 0.45 | 0.41 0.78 0.54 |   51 (113) |  632 (163) |
| 1b. pauses seules, fin de phrase + pause ≥ 400 ms | 0.63 0.48 **0.55** | 0.59 0.45 0.51 | 0.20 0.89 0.33 |   87 (113) |  240 (163) |
| 1b. pauses seules, fin de phrase + pause ≥ 300 ms | 0.61 0.57 **0.59** | 0.57 0.53 0.55 | 0.18 1.00 0.31 |  106 (113) |  191 (163) |
| 2. pauses + indices textuels (q≥0, marqueur≥500) | 0.85 0.46 **0.60** | 0.83 0.44 0.57 | 0.44 0.78 0.56 |   62 (113) |  525 (163) |
| 2a. pauses + question seule        | 0.87 0.43 **0.57** | 0.85 0.40 0.54 | 0.50 0.78 0.61 |   56 (113) |  537 (163) |
| 2b. pauses + marqueurs seuls (≥300 ms) | 0.90 0.41 **0.56** | 0.88 0.37 0.52 | 0.62 0.89 0.73 |   52 (113) |  499 (163) |
| 3. pauses + MFCC+F0 (pyin) (seuil 7.00, pause≥0, vs para, F0×6) | 0.74 0.89 **0.81** | 0.72 0.88 0.79 | 0.20 1.00 0.34 |  136 (113) |  123 (163) |
| 3. pauses + MFCC seuls (seuil 5.00, pause≥0, vs prev, F0×1) | 0.70 0.84 **0.76** | 0.68 0.83 0.75 | 0.17 0.89 0.28 |  135 (113) |  113 (163) |
| 3. pauses + MFCC+F0+écarts-types (seuil 8.75, pause≥0, vs para, F0×6) | 0.73 0.88 **0.80** | 0.71 0.86 0.78 | 0.20 1.00 0.33 |  135 (113) |  105 (163) |
| 3. pauses + MFCC+F0 (YIN simple) (seuil 6.00, pause≥0, vs para, F0×4) | 0.73 0.88 **0.80** | 0.71 0.87 0.79 | 0.20 1.00 0.33 |  136 (113) |  134 (163) |
| 3. pauses + F0 seul (seuil 1.30, pause≥0) | 0.85 0.76 **0.80** | 0.84 0.74 0.78 | 0.38 1.00 0.55 |  101 (113) |  190 (163) |
| 3+2. acoustique (seuil 6.25, pause≥0, F0×3) + indices textuels | 0.79 0.91 **0.85** | 0.78 0.92 0.84 | 0.21 0.78 0.33 |  130 (113) |  138 (163) |
| 4a. pauses + WeSpeaker ResNet34 (cos > 0.60, pause≥0, vs prev) | 0.94 0.88 **0.91** | 0.94 0.87 0.90 | 0.60 1.00 0.75 |  106 (113) |  175 (163) |
| 4b. pauses + segmentation locale PyanNet (changement > 0.3, pause≥0) | 0.86 0.79 **0.82** | 0.84 0.79 0.81 | 0.35 0.89 0.50 |  105 (113) |  205 (163) |
| 4b+2. segmentation locale + indices textuels | 0.81 0.82 **0.81** | 0.79 0.83 0.81 | 0.24 0.78 0.37 |  115 (113) |  195 (163) |

**Koppel (test)** — référence : 75 § (25 par la voix, 49 par les pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane car. (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules (règle bureau)    | 0.82 0.69 **0.75** | 0.45 0.36 0.40 | 0.79 0.86 0.82 |   63 (75) |  696 (493) |
| 1b. pauses seules, fin de phrase + pause ≥ 600 ms | 0.43 0.64 **0.51** | 0.18 0.56 0.28 | 0.35 0.67 0.46 |  110 (75) |  292 (493) |
| 1b. pauses seules, fin de phrase + pause ≥ 400 ms | 0.25 0.78 **0.38** | 0.09 0.64 0.15 | 0.20 0.86 0.32 |  231 (75) |  144 (493) |
| 1b. pauses seules, fin de phrase + pause ≥ 300 ms | 0.22 0.85 **0.35** | 0.07 0.68 0.13 | 0.17 0.94 0.29 |  290 (75) |  110 (493) |
| 2. pauses + indices textuels (q≥0, marqueur≥500) | 0.50 0.76 **0.61** | 0.24 0.68 0.35 | 0.41 0.80 0.55 |  112 (75) |  292 (493) |
| 2a. pauses + question seule        | 0.57 0.76 **0.65** | 0.25 0.56 0.34 | 0.49 0.86 0.63 |  100 (75) |  318 (493) |
| 2b. pauses + marqueurs seuls (≥300 ms) | 0.54 0.68 **0.60** | 0.25 0.56 0.35 | 0.46 0.73 0.57 |   93 (75) |  395 (493) |
| 3. pauses + MFCC+F0 (pyin) (seuil 7.00, pause≥0, vs para, F0×6) | 0.32 0.69 **0.44** | 0.13 0.64 0.21 | 0.24 0.71 0.36 |  161 (75) |  149 (493) |
| 3. pauses + MFCC seuls (seuil 5.00, pause≥0, vs prev, F0×1) | 0.57 0.72 **0.63** | 0.30 0.68 0.41 | 0.47 0.73 0.58 |   94 (75) |  375 (493) |
| 3. pauses + MFCC+F0+écarts-types (seuil 8.75, pause≥0, vs para, F0×6) | 0.38 0.72 **0.50** | 0.16 0.64 0.25 | 0.30 0.76 0.43 |  141 (75) |  169 (493) |
| 3. pauses + MFCC+F0 (YIN simple) (seuil 6.00, pause≥0, vs para, F0×4) | 0.38 0.68 **0.48** | 0.15 0.60 0.24 | 0.30 0.71 0.42 |  134 (75) |  177 (493) |
| 3. pauses + F0 seul (seuil 1.30, pause≥0) | 0.44 0.68 **0.53** | 0.18 0.56 0.27 | 0.36 0.73 0.48 |  115 (75) |  253 (493) |
| 3+2. acoustique (seuil 6.25, pause≥0, F0×3) + indices textuels | 0.38 0.77 **0.51** | 0.18 0.80 0.29 | 0.28 0.76 0.41 |  151 (75) |  197 (493) |
| 4a. pauses + WeSpeaker ResNet34 (cos > 0.60, pause≥0, vs prev) | 0.78 0.78 **0.78** | 0.53 0.72 0.61 | 0.71 0.82 0.76 |   75 (75) |  493 (493) |
| 4b. pauses + segmentation locale PyanNet (changement > 0.3, pause≥0) | 0.85 0.81 **0.83** | 0.59 0.64 0.62 | 0.80 0.90 0.85 |   72 (75) |  583 (493) |
| 4b+2. segmentation locale + indices textuels | 0.51 0.80 **0.62** | 0.27 0.84 0.41 | 0.40 0.78 0.53 |  117 (75) |  270 (493) |