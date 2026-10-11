"""All candidates: thresholds tuned on the first half of the Mel Robbins episode, scored on its
second half and on the Koppel episode (and on the whole of Mel for reference). -I friendly."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import *

mel = load(MEL)
KOPW = os.path.join(HERE, "data", "koppel_whisper.json")        # re-transcribed with word times + pyannote voices
kop = load(KOPW) if os.path.exists(KOPW) else load(KOP, os.path.join(HERE, "data", "koppel_turns.json"))
H = len(mel) // 2
SETS = [("Mel 1re moitié (réglage)", mel, 0, H), ("Mel 2e moitié (test)", mel, H, None), ("Koppel (test)", kop, 0, None)]
results = {}          # name -> {set: score}
preds = {}            # name -> {set index: starts}


def evaluate(name, maker_for, note=""):
    """maker_for(lines_id) -> maker or None (candidate 1)."""
    results[name] = {}
    for si, (sname, lines, lo, hi) in enumerate(SETS):
        mk = maker_for("mel" if lines is mel else "kop")
        s, pred = run(lines, mk, lo, hi)
        results[name][sname] = s
        preds.setdefault(name, {})[si] = pred
    return results[name]


def tune(label, makers_for_mel, makers_for_kop):
    """pick the parameter set with the best F1 on Mel's first half; return (param, maker_for)"""
    best = None
    for key, mk in makers_for_mel.items():
        s, _ = run(mel, mk, 0, H)
        if best is None or s["all"]["F1"] > best[1]:
            best = (key, s["all"]["F1"])
    key = best[0]
    print("  %s: best on tune set = %s (F1 %.2f)" % (label, key, best[1]))
    return key, (lambda which: makers_for_mel[key] if which == "mel" else makers_for_kop[key])


def table(names):
    out = []
    for si, (sname, lines, lo, hi) in enumerate(SETS):
        out.append("\n**%s** — référence : %d § (%d par la voix, %d par les pauses)\n" % (
            sname, results[names[0]][sname]["all"]["n"] + 1, results[names[0]][sname]["voice"]["n"], results[names[0]][sname]["pause"]["n"]))
        out.append(HEADER)
        for n in names:
            out.append(fmt(n, results[n][sname]))
    return "\n".join(out)


# ---- 1. pauses only ----
evaluate("1. pauses seules (règle bureau)", lambda w: None)
# pauses only, thresholds relaxed: is there a pause threshold that catches the voice changes?
for ms in (600, 400, 300):
    def mk(w, ms=ms):
        def make(lines):
            return lambda i, prev, para, pause, ended: pause >= ms and ended
        return make
    evaluate("1b. pauses seules, fin de phrase + pause ≥ %d ms" % ms, mk)

# ---- 2. pauses + text cues ----
print("candidat 2")
variants = {}
for q_ms in (0, 200, 400):
    for m_ms in (0, 200, 300, 500):
        variants[(q_ms, m_ms)] = text_cue("en", q_ms=q_ms, m_ms=m_ms)
key2, mk2 = tune("2 (q_ms, m_ms)", variants, variants)
evaluate("2. pauses + indices textuels (q≥%d, marqueur≥%d)" % key2, mk2)
evaluate("2a. pauses + question seule", lambda w: text_cue("en", q_ms=0, use_m=False))
evaluate("2b. pauses + marqueurs seuls (≥300 ms)", lambda w: text_cue("en", m_ms=300, use_q=False))

# ---- 3. pauses + cheap acoustics ----
fm, fk = os.path.join(HERE, "data", "mel_feat.npz"), os.path.join(HERE, "data", "koppel_feat.npz")
if os.path.exists(fm) and os.path.exists(fk):
    print("candidat 3")
    Fm, Fk = dict(np.load(fm)), dict(np.load(fk))
    ym, yk = os.path.join(HERE, "data", "mel_feat.yin.npz"), os.path.join(HERE, "data", "koppel_feat.yin.npz")
    Ym = dict(Fm, f0=np.load(ym)["f0_yin"]) if os.path.exists(ym) else None
    Yk = dict(Fk, f0=np.load(yk)["f0_yin"]) if os.path.exists(yk) else None
    def f0_only(F):
        f0 = F["f0"].copy(); f0[np.isnan(f0)] = np.nanmedian(f0)
        return ((f0 - f0.mean()) / f0.std())[:, None]
    specs = [("MFCC+F0 (pyin)", dict(use_f0=True), Fm, Fk), ("MFCC seuls", dict(use_f0=False), Fm, Fk),
             ("MFCC+F0+écarts-types", dict(use_f0=True, use_std=True), Fm, Fk)]
    if Ym is not None and Yk is not None:
        specs.append(("MFCC+F0 (YIN simple)", dict(use_f0=True), Ym, Yk))
    for label, kw, Fa, Fb in specs:
        Xm, Xk = acoustic_vectors(Fa, **kw), acoustic_vectors(Fb, **kw)
        vm, vk = {}, {}
        for thr in np.arange(1.5, 9.01, 0.25):
            for gate in (0, 200, 300):
                for against in ("para", "prev"):
                    for f0w in ((1.0, 2.0, 3.0, 4.0, 6.0) if kw.get("use_f0") else (1.0,)):
                        k = (round(float(thr), 2), gate, against, f0w)
                        vm[k] = acoustic_cue(Xm, thr, gate, against=against, f0w=f0w)
                        vk[k] = acoustic_cue(Xk, thr, gate, against=against, f0w=f0w)
        key3, mk3 = tune("3 %s (seuil, porte ms, contre, poids F0)" % label, vm, vk)
        evaluate("3. pauses + %s (seuil %.2f, pause≥%d, vs %s, F0×%g)" % ((label,) + key3), mk3)
    # F0 alone (cheapest of all)
    Xm, Xk = f0_only(Fm), f0_only(Fk)
    vm = {(round(float(t), 2), g, "para", 1.0): acoustic_cue(Xm, t, g, against="para") for t in np.arange(0.5, 3.01, 0.1) for g in (0, 200, 300)}
    vk = {k: acoustic_cue(Xk, k[0], k[1], against="para") for k in vm}
    key3f, mk3f = tune("3 F0 seul (seuil, porte)", vm, vk)
    evaluate("3. pauses + F0 seul (seuil %.2f, pause≥%d)" % key3f[:2], mk3f)
    # 3 + 2 together
    Xm, Xk = acoustic_vectors(Fm), acoustic_vectors(Fk)
    def combo(which, thr=None, gate=None, against=None, f0w=None):
        X = Xm if which == "mel" else Xk
        a = acoustic_cue(X, thr, gate, against=against, f0w=f0w)
        t = text_cue("en", q_ms=key2[0], m_ms=key2[1])
        def make(lines):
            ea, et = a(lines), t(lines)
            return lambda *args: ea(*args) or et(*args)
        return make
    vm = {}
    for thr in np.arange(2.0, 9.01, 0.25):
        for gate in (0, 200, 300):
            for f0w in (1.0, 2.0, 3.0, 4.0, 6.0):
                vm[(round(float(thr), 2), gate, "para", f0w)] = combo("mel", thr, gate, "para", f0w)
    vk = {k: combo("kop", *k) for k in vm}
    key32, mk32 = tune("3+2", vm, vk)
    evaluate("3+2. acoustique (seuil %.2f, pause≥%d, F0×%g) + indices textuels" % (key32[0], key32[1], key32[3]), mk32)
    print("coût MFCC sur tout le fichier (librosa, bureau) : %.4f × temps réel" % (float(Fm["t_mfcc"]) / float(Fm["audio_s"])))

# ---- 4. embeddings / local segmentation ----
em, ek = os.path.join(HERE, "data", "mel_emb.npz"), os.path.join(HERE, "data", "koppel_emb.npz")
if os.path.exists(em) and os.path.exists(ek):
    print("candidat 4")
    Em, Ek = np.load(em), np.load(ek)
    vm, vk = {}, {}
    for thr in np.arange(0.2, 0.9, 0.05):
        for gate in (0, 200, 300):
            for against in ("para", "prev"):
                k = (round(float(thr), 2), gate, against)
                vm[k] = embedding_cue(Em["emb"], thr, gate, against)
                vk[k] = embedding_cue(Ek["emb"], thr, gate, against)
    key4, mk4 = tune("4a WeSpeaker (seuil cos, porte, contre)", vm, vk)
    evaluate("4a. pauses + WeSpeaker ResNet34 (cos > %.2f, pause≥%d, vs %s)" % key4, mk4)
    vm, vk = {}, {}
    for thr in (0.3, 0.5, 0.7, 0.8, 0.9):
        for gate in (0, 200, 300):
            vm[(thr, gate)] = segmentation_cue(Em["seg_before"], Em["seg_after"], Em["seg_change"], thr, gate)
            vk[(thr, gate)] = segmentation_cue(Ek["seg_before"], Ek["seg_after"], Ek["seg_change"], thr, gate)
    key4b, mk4b = tune("4b PyanNet segmentation locale (seuil, porte)", vm, vk)
    evaluate("4b. pauses + segmentation locale PyanNet (changement > %.1f, pause≥%d)" % key4b, mk4b)
    # 4b + 2
    def combo4(which):
        E = Em if which == "mel" else Ek
        s = segmentation_cue(E["seg_before"], E["seg_after"], E["seg_change"], key4b[0], key4b[1])
        t = text_cue("en", q_ms=key2[0], m_ms=key2[1])
        def make(lines):
            es, et = s(lines), t(lines)
            return lambda *args: es(*args) or et(*args)
        return make
    evaluate("4b+2. segmentation locale + indices textuels", combo4)
    print("coût WeSpeaker (1 fil, bureau) : %.3f × temps réel ; segmentation : %.3f × durée des fenêtres, %.3f × durée audio" % (
        float(Em["t_emb"]) / float(Em["emb_secs"]), float(Em["t_seg"]) / float(Em["seg_secs"]), float(Em["t_seg"]) / float(Em["audio_s"])))

names = list(results)
txt = table(names)
print(txt)
open(os.path.join(HERE, "data", "table.md"), "w").write(txt)
json.dump({n: {si: sorted(p) for si, p in d.items()} for n, d in preds.items()}, open(os.path.join(HERE, "data", "preds.json"), "w"))
