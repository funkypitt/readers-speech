"""Passages where the candidates differ from the reference, from data/preds.json.  examples.py [names...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import *

EP = os.environ.get("EP", "mel")
mel = load(MEL) if EP == "mel" else load(os.path.join(HERE, "data", "koppel_whisper.json"))
ref, kind = desktop(mel, voices=True)
preds = json.load(open(os.path.join(HERE, "data", "preds.json")))
names = [n for n in preds if any(n.startswith(p) for p in (sys.argv[1:] or ("1.", "2.", "3.", "4b.")))]
P = {n: (set(preds[n]["0"]) | set(preds[n]["1"])) if EP == "mel" else set(preds[n]["2"]) for n in names}
short = {n: n.split(".")[0] + ("." + n.split(".")[1][:1] if n[1] != "." else "") for n in names}
short = {n: n.split(" ")[0].rstrip(".") for n in names}


def show(i, why):
    print("\n--- ligne %d — %s" % (i, why))
    for j in range(max(0, i - 2), min(len(mel), i + 2)):
        l = mel[j]
        marks = "".join(("[%s]" % short[n]) if j in P[n] else "" for n in names)
        print("%s%4d v%s +%4d ms %-28s %s" % ("¶ " if j in ref else "  ", j, l["s"], l["a"] - mel[j - 1]["b"] if j else 0, marks, l["t"]))


def pick(cond, k, why):
    hits = [i for i in sorted(ref) if kind[i] != "first" and cond(i)]
    print("\n==== %s : %d cas" % (why, len(hits)))
    for i in hits[:k]:
        show(i, why)
    return hits


q = lambda i: mel[i - 1]["t"].rstrip().endswith("?")
n1, n2 = names[0], names[1] if len(names) > 1 else names[0]
pick(lambda i: kind[i] == "voice" and q(i) and i not in P[n1], 3, "question puis réponse, manquée par 1, vue par 2")
pick(lambda i: kind[i] == "voice" and not q(i) and i not in P[n2] and any(i in P[n] for n in names[2:]), 4,
     "changement de voix sans ?, manqué par 2, vu par 3 ou 4")
pick(lambda i: kind[i] == "voice" and all(i not in P[n] for n in names), 4, "changement de voix manqué par tous")
# false positives
for n in names:
    fps = [i for i in sorted(P[n]) if i not in ref]
    print("\n==== faux positifs de %s : %d" % (n, len(fps)))
    for i in fps[:3]:
        show(i, "faux positif de " + short[n])
