# Couper une transcription en paragraphes sur le téléphone, sans pyannote

Étude du 2026-10-11. Scripts et données dans ce dossier (`score.py`, `evaluate.py`, `reverse.py`, `features.py`, `embed.py`, `transcribe_koppel.py`, `table.md`, `examples_*.txt`, `preds.json`).

## 1. La question

Le bureau (`readers-podcasts-desktop`, `paragraph_starts`) ouvre un paragraphe quand **la voix change** (pyannote), après une **pause ≥ 2000 ms**, après une pause ≥ 900 ms **en fin de phrase**, ou en fin de phrase avec pause ≥ 250 ms quand le paragraphe dépasse 900 caractères. Le téléphone n'a pas les voix : `Paragraphs.build` (`whisper/Prompts.kt`) ne connaît que les pauses (1,2 s + 250 car., ou 600 car.), et `Transcripts.asText` de Podcasts coupe sur une pause > 1,2 s.

## 2. Données et référence

| épisode | durée | lignes | voix (pyannote) | § de référence | dont voix / pauses |
|---|---|---|---|---|---|
| **Mel Robbins** « The Ultimate Men's Health Guide », transcrit par le bureau | 69 min | 960 | femme 180 Hz, homme 100 Hz | 226 | 202 / 23 |
| **Roger Koppel** | 51 min | 624 | deux hommes, 113 Hz et 137 Hz | 75 | 25 / 49 |

- Le fichier Koppel du bureau n'avait ni voix ni vraies pauses : retranscrit avec faster-whisper `small` (le modèle du téléphone), mots datés, diarisé avec pyannote community-1, lignes coupées au changement de voix comme le bureau le fait. Jeu plus dur : deux voix masculines proches, un invité qui monologue, un hôte qui fait de courtes relances.
- **Référence** = `paragraph_starts` avec les voix. Frontière = indice d'une ligne qui ouvre un paragraphe ; « voix » si la voix change là, « pause » sinon. Score strict (même indice).
- **Réglage / test** : seuils choisis sur la première moitié de Mel, appliqués tels quels à la seconde moitié et à Koppel ; `reverse.py` fait l'inverse.

## 3. Candidats

1. **Pauses seules** : la règle du bureau sans voix. Variante 1b : fin de phrase + pause ≥ 300/400/600 ms.
2. **Pauses + indices textuels** : nouvelle ligne après une ligne finissant par `?` ; ligne commençant par un marqueur de prise de parole (*Yeah, Yes, No, So, Well, Right, Okay, I mean, Absolutely, Exactly, Oh, Mm, Thank you, Sure…* + listes fr/de/es/pt dans `score.py`) après une pause ≥ 500 ms.
3. **Pauses + acoustique bon marché** par ligne : moyenne de 13 MFCC et log-F0 moyen, z-score par épisode, distance à la moyenne du paragraphe courant (ou à la ligne précédente) > seuil.
4. **Petits modèles de pyannote community-1, isolés, CPU 1 fil** :
   - 4a. **WeSpeaker ResNet34** (6,6 M paramètres, 26 Mo) : un embedding par ligne, cosinus à la ligne précédente > seuil.
   - 4b. **PyanNet, le modèle de segmentation** (1,47 M paramètres, 5,7 Mo) : fenêtre de 10 s centrée sur chaque trou entre deux lignes, locuteur local dominant 2 s avant vs 2 s après. Un changement **local**, pas une identité.

## 4. Résultats

P R F1 des frontières ; « voix »/« pauses » = frontières de la référence de chaque sorte ; § = paragraphes (référence entre parenthèses) ; médiane = longueur médiane en caractères.

**Mel 1re moitié (réglage)** — référence : 114 § (99 voix, 14 pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules (règle bureau) | 0.84 0.34 **0.48** | 0.77 0.24 0.37 | 0.67 1.00 0.80 | 46 (114) | 819 (224) |
| 1b. fin de phrase + pause ≥ 600 ms | 0.83 0.44 **0.58** | 0.78 0.36 0.50 | 0.58 1.00 0.74 | 61 | 514 |
| 1b. fin de phrase + pause ≥ 300 ms | 0.71 0.70 **0.71** | 0.67 0.66 0.66 | 0.30 1.00 0.47 | 112 | 193 |
| 2. pauses + texte | 0.91 0.54 **0.68** | 0.89 0.47 0.62 | 0.70 1.00 0.82 | 68 | 440 |
| 2a. question seule | 0.94 0.52 **0.67** | 0.92 0.45 0.61 | 0.78 1.00 0.88 | 64 | 456 |
| 2b. marqueurs seuls | 0.80 0.42 **0.55** | 0.74 0.35 0.48 | 0.52 0.93 0.67 | 61 | 514 |
| 3. MFCC+F0 pyin | 0.89 0.98 **0.93** | 0.87 0.98 0.92 | 0.50 1.00 0.67 | 126 | 184 |
| 3. MFCC seuls | 0.81 0.81 **0.81** | 0.78 0.78 0.78 | 0.39 1.00 0.56 | 114 | 188 |
| 3. MFCC+F0 YIN simple | 0.88 0.96 **0.92** | 0.86 0.95 0.90 | 0.48 1.00 0.65 | 124 | 188 |
| 3. F0 seul | 0.93 0.90 **0.91** | 0.92 0.89 0.90 | 0.64 1.00 0.78 | 111 | 240 |
| 3+2. acoustique + texte | 0.92 0.97 **0.94** | 0.91 0.97 0.94 | 0.58 1.00 0.74 | 121 | 191 |
| 4a. WeSpeaker | 1.00 0.98 **0.99** | 1.00 0.98 0.99 | 1.00 1.00 1.00 | 112 | 233 |
| 4b. segmentation locale PyanNet | 0.97 0.98 **0.97** | 0.96 0.98 0.97 | 0.78 1.00 0.88 | 116 | 203 |
| 4b+2. segmentation locale + texte | 0.93 0.98 **0.95** | 0.92 0.98 0.95 | 0.61 1.00 0.76 | 121 | 191 |

**Mel 2e moitié (test)** — référence : 113 § (103 voix, 9 pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules | 0.85 0.29 **0.44** | 0.80 0.23 0.36 | 0.60 1.00 0.75 | 40 (113) | 957 (163) |
| 1b. fin de phrase + pause ≥ 600 ms | 0.80 0.36 **0.49** | 0.77 0.32 0.45 | 0.41 0.78 0.54 | 51 | 632 |
| 1b. fin de phrase + pause ≥ 300 ms | 0.61 0.57 **0.59** | 0.57 0.53 0.55 | 0.18 1.00 0.31 | 106 | 191 |
| 2. pauses + texte | 0.85 0.46 **0.60** | 0.83 0.44 0.57 | 0.44 0.78 0.56 | 62 | 525 |
| 2a. question seule | 0.87 0.43 **0.57** | 0.85 0.40 0.54 | 0.50 0.78 0.61 | 56 | 537 |
| 3. MFCC+F0 pyin | 0.74 0.89 **0.81** | 0.72 0.88 0.79 | 0.20 1.00 0.34 | 136 | 123 |
| 3. MFCC seuls | 0.70 0.84 **0.76** | 0.68 0.83 0.75 | 0.17 0.89 0.28 | 135 | 113 |
| 3. MFCC+F0 YIN simple | 0.73 0.88 **0.80** | 0.71 0.87 0.79 | 0.20 1.00 0.33 | 136 | 134 |
| 3. F0 seul | 0.85 0.76 **0.80** | 0.84 0.74 0.78 | 0.38 1.00 0.55 | 101 | 190 |
| 3+2. acoustique + texte | 0.79 0.91 **0.85** | 0.78 0.92 0.84 | 0.21 0.78 0.33 | 130 | 138 |
| 4a. WeSpeaker | 0.94 0.88 **0.91** | 0.94 0.87 0.90 | 0.60 1.00 0.75 | 106 | 175 |
| 4b. segmentation locale PyanNet | 0.86 0.79 **0.82** | 0.84 0.79 0.81 | 0.35 0.89 0.50 | 105 | 205 |
| 4b+2. | 0.81 0.82 **0.81** | 0.79 0.83 0.81 | 0.24 0.78 0.37 | 115 | 195 |

**Koppel (test)** — référence : 75 § (25 voix, 49 pauses)

| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane (réf.) |
|---|---|---|---|---|---|
| 1. pauses seules | 0.82 0.69 **0.75** | 0.45 0.36 0.40 | 0.79 0.86 0.82 | 63 (75) | 696 (493) |
| 1b. fin de phrase + pause ≥ 600 ms | 0.43 0.64 **0.51** | 0.18 0.56 0.28 | 0.35 0.67 0.46 | 110 | 292 |
| 1b. fin de phrase + pause ≥ 300 ms | 0.22 0.85 **0.35** | 0.07 0.68 0.13 | 0.17 0.94 0.29 | 290 | 110 |
| 2. pauses + texte | 0.50 0.76 **0.61** | 0.24 0.68 0.35 | 0.41 0.80 0.55 | 112 | 292 |
| 2a. question seule | 0.57 0.76 **0.65** | 0.25 0.56 0.34 | 0.49 0.86 0.63 | 100 | 318 |
| 3. MFCC+F0 pyin | 0.32 0.69 **0.44** | 0.13 0.64 0.21 | 0.24 0.71 0.36 | 161 | 149 |
| 3. MFCC seuls | 0.57 0.72 **0.63** | 0.30 0.68 0.41 | 0.47 0.73 0.58 | 94 | 375 |
| 3. MFCC+F0 YIN simple | 0.38 0.68 **0.48** | 0.15 0.60 0.24 | 0.30 0.71 0.42 | 134 | 177 |
| 3. F0 seul | 0.44 0.68 **0.53** | 0.18 0.56 0.27 | 0.36 0.73 0.48 | 115 | 253 |
| 3+2. acoustique + texte | 0.38 0.77 **0.51** | 0.18 0.80 0.29 | 0.28 0.76 0.41 | 151 | 197 |
| 4a. WeSpeaker | 0.78 0.78 **0.78** | 0.53 0.72 0.61 | 0.71 0.82 0.76 | 75 | 493 |
| 4b. segmentation locale PyanNet | 0.85 0.81 **0.83** | 0.59 0.64 0.62 | 0.80 0.90 0.85 | 72 | 583 |
| 4b+2. | 0.51 0.80 **0.62** | 0.27 0.84 0.41 | 0.40 0.78 0.53 | 117 | 270 |

**Robustesse** (`reverse.py`) : réglage dans l'autre sens, et plafond si réglé sur le test lui-même.

| candidat | réglé sur Koppel → Mel entier | réglé sur Mel-1re → Koppel | plafonds (Koppel ; Mel-2e) |
|---|---|---|---|
| 3 MFCC+F0 | F1 **0.60** | F1 **0.44** | 0.78 ; 0.86 |
| 3 MFCC seuls | F1 **0.54** | F1 **0.63** | 0.79 ; 0.76 |
| 4a WeSpeaker | F1 **0.93** | F1 **0.78** | 0.84 ; 0.91 |
| 4b PyanNet local | F1 **0.86** | F1 **0.83** | 0.83 ; 0.82 |

Variantes de 4b : seuil 0,3-0,7 et porte « fin de phrase ou pause ≥ 200 ms » ne changent F1 que de ±0,02 ; la porte gagne 3 points de précision, évite les coupures en pleine phrase, et il n'y a plus de fenêtre à calculer que sur la moitié des trous.

### Ce que les chiffres disent

- **Pauses seules** (le téléphone d'aujourd'hui) : sur une interview, un quart des changements de voix retrouvés, paragraphes de 800-950 caractères de médiane au lieu de 160-220, la question collée à sa réponse. Sur un monologue à relances la règle du bureau sans voix s'en tire (0,75).
- **Baisser le seuil de pause** rattrape des voix sur Mel mais explose dès que quelqu'un parle lentement (Koppel : 290 § pour 75).
- **Texte** : la question marche sur Mel, les marqueurs n'apportent presque rien ; Koppel pose des questions rhétoriques : F1 global sous les pauses seules. À écarter.
- **MFCC + F0** : brillant femme/homme (c'est la hauteur qui fait tout), mais le seuil ne se transporte pas et deux hommes à 24 Hz d'écart ne se séparent pas. À écarter.
- **4b** : le seul qui tient dans les deux sens, nombre de paragraphes juste, médiane proche. Ses manques sont les relances d'un ou deux mots prises dans une ligne de l'autre voix.
- **4a** : meilleur sur Mel, moins bon sur Koppel, 4,5 × plus lourd. Pas justifié.

## 5. Passages

`¶` = frontière de référence ; `v0/v1` = voix ; `+ms` = pause avant la ligne ; crochets = qui a coupé (1 pauses, 2 texte, 3 F0 seul, 4b segmentation locale).

```
    56 v0 + 360 ms [2]          And what do you want us to know about it when you think about the bigger picture
    57 v0 +   0 ms              before we dive into specific questions?
¶   58 v1 + 680 ms [2][3][4b]   Yeah, so I think it's just sort of optimum hormone levels.

    23 v0 +   0 ms              if they take everything to heart that you're about to teach us?
    24 v0 + 140 ms [2]          What are we going to learn?
¶   25 v1 + 680 ms [2][4b]      Well, I think we're going to learn quite a bit.

    66 v1 + 240 ms              quality of life things, overall health things, functional things.
    67 v1 + 720 ms [2]          So I think that having that understanding, I think is really fundamental.
¶   68 v0 + 360 ms [1][3][4b]   Okay. What age do men start to see changes in testosterone?

   109 v0 +   0 ms              How do you see this, I guess, narrative trap of the Iran war and the Ukraine war?
¶  110 v1 + 720 ms [2][3]       It's the inability to differentiate also.
   228 v0 + 220 ms [2]          Because now everyone has some flaws to them, I think.
¶  229 v1 + 240 ms [4b]         I try to look at these issues, not so much, for example, in terms of values and all those words.
```

## 6. Coût sur le téléphone

Mesures sur un cœur d'i7-12700K, un fil ; un gros cœur de téléphone est 2-3 × plus lent. Repère : whisper `small` ≈ 0,5 × temps réel sur le téléphone.

| candidat | bureau, 1 fil | téléphone (estimation) | par heure de son | taille |
|---|---|---|---|---|
| 1-2 pauses, texte | 0 | 0 | 0 | 0 |
| 3 MFCC par ligne | 0,0006 × | 0,002-0,004 × | ~10 s | 0 |
| 3 F0 YIN simple | 0,003-0,006 × | 0,01-0,03 × | 1-2 min | 0 |
| 4a WeSpeaker | 0,04 × | 0,1-0,2 × | 6-12 min | 26 Mo |
| 4b PyanNet, une fenêtre de 10 s par trou | 0,009 × | **0,01-0,03 ×** avec la porte | **1-3 min** | **5,7 Mo** |

4b coûte 2-6 % du temps de whisper `small`. Pyannote complet sur téléphone : des dizaines de minutes et 32 Mo.

## 7. Recommandation

**Implémenter 4b dans readers-speech : le modèle de segmentation de pyannote (PyanNet, 1,5 M paramètres) en ONNX, appliqué localement autour de chaque trou entre deux segments whisper, pendant que le morceau de son est encore en mémoire.** Ni texte ni MFCC/F0.

- **Modèle** : sherpa-onnx redistribue `sherpa-onnx-pyannote-segmentation-3-0.onnx` (≈ 6 Mo, MIT) ; sinon exporter `pipe._segmentation.model` de community-1 (licence à vérifier). Exécution par `onnxruntime-android` (dépendance nouvelle, ~10-15 Mo par ABI).
- **Où** : `whisper/WhisperLib.kt`. `WhisperSession.run(pcm16k, …)` a sous la main le `FloatArray` du morceau de 300 s et les segments à 10 ms. Juste après `fullTranscribe`, un nouvel objet `whisper/Voices.kt` : pour chaque trou (si le segment précédent finit une phrase ou pause ≥ 200 ms), fenêtre de 10 s centrée sur le trou, locuteur local dominant 2 s avant / 2 s après ; différents → marquer. `data class Segment` gagne `newVoice: Boolean = false`.
- **Règle** : `Paragraphs.build` coupe quand `s.newVoice` ; aligner les seuils de pause sur ceux du bureau (2000 ; 900 en fin de phrase ; 900 car. + 250 ms).
- **Podcasts** : `Transcripts.fromSegments` garde le drapeau dans `Line`, `write` l'écrit comme `"p": 1` — le bureau le lit déjà.
- **Repli** sans modèle : la règle du bureau sans voix, pas les indices textuels.
- **À mesurer avant de publier** : le temps réel d'une fenêtre sur un vrai téléphone, et que l'export ONNX redonne les mêmes frontières que PyTorch sur les deux épisodes (`score.py`, `preds.json`).

Attendu : sur une interview, F1 ≈ 0,8-0,9 contre 0,45 aujourd'hui ; sur un monologue à relances, 0,83 contre 0,75 ; pour une à trois minutes de calcul par heure de son.
