"""Robustness check: tune on Koppel, test on Mel (whole), plus oracle (best-on-test) ceilings and
a ±1-line tolerance, for the acoustic candidates."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import *

mel = load(MEL)
kop = load(os.path.join(HERE, "data", "koppel_whisper.json"))
Fm, Fk = dict(np.load(os.path.join(HERE, "data", "mel_feat.npz"))), dict(np.load(os.path.join(HERE, "data", "koppel_feat.npz")))
Em, Ek = np.load(os.path.join(HERE, "data", "mel_emb.npz")), np.load(os.path.join(HERE, "data", "koppel_emb.npz"))


def best(lines, makers, lo=0, hi=None):
    out = None
    for k, mk in makers.items():
        s, _ = run(lines, mk, lo, hi)
        if out is None or s["all"]["F1"] > out[1]["all"]["F1"]:
            out = (k, s)
    return out


def both(label, mk_mel, mk_kop):
    """tune on Koppel -> Mel ; tune on Mel(1st half) -> Koppel ; oracle on each"""
    kk, sk = best(kop, mk_kop)
    sm, _ = run(mel, mk_mel[kk])
    km, _ = best(mel, mk_mel, 0, len(mel) // 2)
    sk2, _ = run(kop, mk_kop[km])
    om = best(mel, mk_mel, len(mel) // 2)
    print("| %-32s | %s : F1 Mel %.2f (voix %.2f) | %s : F1 Koppel %.2f (voix %.2f) | plafond Koppel %.2f, Mel-2e %.2f |" % (
        label, kk, sm["all"]["F1"], sm["voice"]["F1"], km, sk2["all"]["F1"], sk2["voice"]["F1"], sk["all"]["F1"], om[1]["all"]["F1"]))


print("| candidat | réglé sur Koppel → Mel | réglé sur Mel-1re → Koppel | plafonds (réglé sur le test lui-même) |")
print("|---|---|---|---|")
for label, kw in (("3 MFCC+F0", dict(use_f0=True)), ("3 MFCC seuls", dict(use_f0=False))):
    Xm, Xk = acoustic_vectors(Fm, **kw), acoustic_vectors(Fk, **kw)
    keys = [(round(float(t), 2), g, ag, w) for t in np.arange(1.5, 9.01, 0.25) for g in (0, 200, 300) for ag in ("para", "prev")
            for w in ((1.0, 2.0, 3.0, 4.0, 6.0) if kw["use_f0"] else (1.0,))]
    both(label, {k: acoustic_cue(Xm, k[0], k[1], against=k[2], f0w=k[3]) for k in keys},
         {k: acoustic_cue(Xk, k[0], k[1], against=k[2], f0w=k[3]) for k in keys})
keys = [(round(float(t), 2), g, ag) for t in np.arange(0.2, 0.9, 0.05) for g in (0, 200, 300) for ag in ("para", "prev")]
both("4a WeSpeaker", {k: embedding_cue(Em["emb"], *k) for k in keys}, {k: embedding_cue(Ek["emb"], *k) for k in keys})
keys = [(t, g) for t in (0.3, 0.5, 0.7, 0.8, 0.9) for g in (0, 200, 300)]
both("4b PyanNet local", {k: segmentation_cue(Em["seg_before"], Em["seg_after"], Em["seg_change"], *k) for k in keys},
     {k: segmentation_cue(Ek["seg_before"], Ek["seg_after"], Ek["seg_change"], *k) for k in keys})

print("\nTolérance ±1 ligne (Mel entier / Koppel) :")
for label, mm, mk in (("1 pauses", None, None),
                      ("2 texte", text_cue("en", 0, 500), text_cue("en", 0, 500)),
                      ("4b PyanNet (0.3, 0)", segmentation_cue(Em["seg_before"], Em["seg_after"], Em["seg_change"], 0.3, 0),
                       segmentation_cue(Ek["seg_before"], Ek["seg_after"], Ek["seg_change"], 0.3, 0)),
                      ("4a WeSpeaker (0.6, prev)", embedding_cue(Em["emb"], 0.6, 0, "prev"), embedding_cue(Ek["emb"], 0.6, 0, "prev"))):
    a0, _ = run(mel, mm); a1, _ = run(mel, mm, tol=1); b0, _ = run(kop, mk); b1, _ = run(kop, mk, tol=1)
    print("  %-26s Mel F1 %.2f → %.2f ; Koppel F1 %.2f → %.2f" % (label, a0["all"]["F1"], a1["all"]["F1"], b0["all"]["F1"], b1["all"]["F1"]))
