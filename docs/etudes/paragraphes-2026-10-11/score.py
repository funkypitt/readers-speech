"""Paragraph boundaries on the phone, without pyannote: candidates scored against the desktop's
rule with voices. Plain Python + numpy; run with -I.  score.py [--examples]"""
import json, os, re, sys, statistics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.expanduser("~/.local/share/readers-podcasts/transcripts/")
MEL, KOP = "6f08828f8ce2815f744525a5d9343c4c1eac1d57", "9cf4f24132f0da6df20b05b45d3c6bab38c17893"
_SENTENCE_END = tuple(".?!…»”\"")


# ---------------- data ----------------
def voice_over(turns, a, b):
    share = {}
    for ta, tb, v in turns:
        if tb <= a:
            continue
        if ta >= b:
            break
        share[v] = share.get(v, 0) + min(b, tb) - max(a, ta)
    return max(share, key=share.get) if share else None


def load(eid, turns_file=None):
    path = eid if os.sep in eid else TR + eid + ".json"
    rows = json.load(open(path))["lines"]
    lines = [dict(a=int(r["a"]), b=int(r["b"]), t=r["t"], s=r.get("s")) for r in rows]
    if turns_file:
        d = json.load(open(turns_file))
        turns = sorted((int(t["a"] * 1000), int(t["b"] * 1000), t["s"]) for t in d["turns"])
        numbers = {}
        for l in lines:
            v = voice_over(turns, l["a"], l["b"])
            l["s"] = None if v is None else numbers.setdefault(v, len(numbers))
    return lines


# ---------------- the desktop rule (reference when `voices`), candidate 1 when not ----------------
def desktop(lines, voices=True, extra=None, long_ms=2000, sent_ms=900, len_chars=900, len_ms=250):
    """Returns (starts, kind) — kind[i] = 'voice' | 'pause' | 'extra'. `extra(i, prev, paragraph_lines)`
    may add a boundary of its own (candidates 2-4)."""
    starts, kind, length, prev, para = set(), {}, 0, None, []
    for i, l in enumerate(lines):
        t = l["t"]
        if not t.strip():
            continue
        if prev is None:
            starts.add(i); kind[i] = "first"
        else:
            p = lines[prev]
            pause, ended = l["a"] - p["b"], p["t"].rstrip().endswith(_SENTENCE_END)
            other = voices and l["s"] is not None and p["s"] is not None and l["s"] != p["s"]
            by_pause = pause >= long_ms or (pause >= sent_ms and ended) or (length >= len_chars and ended and pause >= len_ms)
            if other or by_pause:
                starts.add(i); kind[i] = "voice" if other else "pause"
                length, para = 0, []
            elif extra is not None and extra(i, prev, para, pause, ended):
                starts.add(i); kind[i] = "extra"
                length, para = 0, []
        length += len(t)
        para.append(i)
        prev = i
    return starts, kind


# ---------------- candidate 2: text cues ----------------
MARKERS = {
    "en": ["yeah", "yes", "no", "so", "well", "right", "okay", "ok", "i mean", "absolutely", "exactly", "oh", "mm", "mhm",
           "hmm", "thank you", "thanks", "sure", "great", "wow", "totally", "correct", "that's right", "of course", "true"],
    "fr": ["oui", "non", "alors", "bon", "ben", "eh bien", "d'accord", "exactement", "absolument", "tout à fait", "voilà",
           "ah", "oh", "merci", "bien sûr", "en fait", "effectivement", "c'est ça", "ouais", "hmm", "euh", "donc", "écoutez"],
    "de": ["ja", "nein", "also", "gut", "genau", "richtig", "okay", "absolut", "danke", "natürlich", "ach", "oh", "hm",
           "stimmt", "klar", "naja", "nun", "doch", "eben"],
    "es": ["sí", "no", "bueno", "pues", "claro", "exacto", "exactamente", "vale", "gracias", "ah", "oh", "absolutamente",
           "por supuesto", "es verdad", "totalmente", "mira"],
    "pt": ["sim", "não", "bem", "bom", "então", "claro", "exato", "exatamente", "obrigado", "obrigada", "pois", "ah", "oh",
           "com certeza", "olha"],
}
_MARK_RE = {lang: re.compile(r"^(?:%s)\b" % "|".join(re.escape(m) for m in sorted(ms, key=len, reverse=True)), re.I)
            for lang, ms in MARKERS.items()}


def text_cue(lang, q_ms=0, m_ms=300, use_q=True, use_m=True):
    rx = _MARK_RE[lang]

    def extra(i, prev, para, pause, ended, lines=None):
        return False
    def make(lines):
        def extra(i, prev, para, pause, ended):
            pt = lines[prev]["t"].rstrip()
            if use_q and pt.endswith(("?", "?\"", "?»", "?”")) and pause >= q_ms:
                return True
            if use_m and pause >= m_ms and rx.match(lines[i]["t"].strip()):
                return True
            return False
        return extra
    return make


# ---------------- candidate 3: cheap acoustics ----------------
def acoustic_vectors(feat, use_f0=True, drop_c0=True, use_std=False):
    m = feat["mfcc_mean"][:, 1:] if drop_c0 else feat["mfcc_mean"]
    cols = [m]
    if use_std:
        cols.append(feat["mfcc_std"][:, 1:])
    if use_f0:
        f0 = feat["f0"].copy()
        f0[np.isnan(f0)] = np.nanmedian(f0)
        cols.append(f0[:, None])
    X = np.concatenate(cols, 1)
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)           # per-episode z-score (phone: running stats)
    return X


def acoustic_cue(X, thr, gate_ms=0, need_end=False, against="para", f0w=1.0):
    W = np.ones(X.shape[1]); W[-1] = f0w
    def make(lines):
        def extra(i, prev, para, pause, ended):
            if pause < gate_ms or (need_end and not ended):
                return False
            ref = X[para].mean(0) if against == "para" and para else X[prev]
            return float(np.sqrt((((X[i] - ref) * W) ** 2).sum())) > thr
        return extra
    return make


# ---------------- candidate 4: embeddings / local segmentation ----------------
def cos_dist(u, v):
    return 1.0 - float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def embedding_cue(E, thr, gate_ms=0, against="para"):
    def make(lines):
        def extra(i, prev, para, pause, ended):
            if pause < gate_ms:
                return False
            ref = E[para].mean(0) if against == "para" and para else E[prev]
            return cos_dist(E[i], ref) > thr
        return extra
    return make


def segmentation_cue(before, after, change, thr=0.5, gate_ms=0):
    def make(lines):
        def extra(i, prev, para, pause, ended):
            if pause < gate_ms:
                return False
            return before[i] >= 0 and after[i] >= 0 and before[i] != after[i] and change[i] > thr
        return extra
    return make


# ---------------- scoring ----------------
def score(pred, ref, kind, lo=0, hi=None, tol=0):
    hi = hi if hi is not None else 10 ** 9
    P = {i for i in pred if lo <= i < hi and kind.get(i) != "first"}
    R = {i for i in ref if lo <= i < hi and kind.get(i) != "first"}
    def near(i, S):
        return any(j in S for j in range(i - tol, i + tol + 1))
    hits = {i for i in R if near(i, P)}
    fp = len([i for i in P if not near(i, ref)])
    out = {}
    for k in ("voice", "pause"):
        Rk = {i for i in R if kind[i] == k}
        hk = len(hits & Rk)
        p = hk / (hk + fp) if hk + fp else 0.0
        r = hk / len(Rk) if Rk else 0.0
        out[k] = dict(P=p, R=r, F1=2 * p * r / (p + r) if p + r else 0.0, n=len(Rk))
    h = len(hits)
    p = h / len(P) if P else 0.0
    r = h / len(R) if R else 0.0
    out["all"] = dict(P=p, R=r, F1=2 * p * r / (p + r) if p + r else 0.0, n=len(R), npred=len(P))
    return out


def para_stats(lines, starts, lo=0, hi=None):
    hi = hi if hi is not None else len(lines)
    lens, cur = [], 0
    for i in range(lo, hi):
        if not lines[i]["t"].strip():
            continue
        if i in starts and cur:
            lens.append(cur); cur = 0
        cur += len(lines[i]["t"].strip()) + 1
    if cur:
        lens.append(cur)
    return len(lens), statistics.median(lens) if lens else 0


def run(lines, make, lo=0, hi=None, voices=False, tol=0, **kw):
    ref, kind = desktop(lines, voices=True)
    pred, _k = desktop(lines, voices=voices, extra=make(lines) if make else None, **kw)
    s = score(pred, ref, kind, lo, hi, tol)
    s["paras"], s["median"] = para_stats(lines, pred, lo, hi)
    s["ref_paras"], s["ref_median"] = para_stats(lines, ref, lo, hi)
    return s, pred


def fmt(name, s):
    a, v, p = s["all"], s["voice"], s["pause"]
    return "| %-34s | %4.2f %4.2f **%4.2f** | %4.2f %4.2f %4.2f | %4.2f %4.2f %4.2f | %4d (%d) | %4d (%d) |" % (
        name, a["P"], a["R"], a["F1"], v["P"], v["R"], v["F1"], p["P"], p["R"], p["F1"], s["paras"], s["ref_paras"], s["median"], s["ref_median"])


HEADER = ("| candidat | tous P R F1 | voix P R F1 | pauses P R F1 | § (réf.) | médiane car. (réf.) |\n"
          "|---|---|---|---|---|---|")


def grid(lines, lo, hi, makers):
    """best (F1 on [lo,hi)) among {name: maker}"""
    best = None
    for name, mk in makers.items():
        s, _ = run(lines, mk, lo, hi)
        if best is None or s["all"]["F1"] > best[1]["all"]["F1"]:
            best = (name, s, mk)
    return best


if __name__ == "__main__":
    mel = load(MEL)
    kop = load(KOP, os.path.join(HERE, "data", "koppel_turns.json"))
    ref_m, kind_m = desktop(mel, voices=True)
    ref_k, kind_k = desktop(kop, voices=True)
    import collections
    print("Mel: lines", len(mel), "ref starts", len(ref_m), collections.Counter(kind_m.values()))
    print("Koppel: lines", len(kop), "ref starts", len(ref_k), collections.Counter(kind_k.values()),
          "voices", collections.Counter(l["s"] for l in kop))
    pauses = [mel[i]["a"] - mel[i - 1]["b"] for i in range(1, len(mel))]
    print("Mel pauses: median %d, >=300: %d, >=900: %d, >=2000: %d" % (statistics.median(pauses), sum(p >= 300 for p in pauses),
          sum(p >= 900 for p in pauses), sum(p >= 2000 for p in pauses)))
    vc = [i for i in ref_m if kind_m[i] == "voice"]
    print("Mel voice-change boundaries: pause median %d ms; ended-sentence before: %d/%d; prev ends '?': %d" % (
        statistics.median(mel[i]["a"] - mel[i - 1]["b"] for i in vc),
        sum(mel[i - 1]["t"].rstrip().endswith(_SENTENCE_END) for i in vc), len(vc),
        sum(mel[i - 1]["t"].rstrip().endswith("?") for i in vc)))
    print("Mel: lines whose text ends with '?':", sum(l["t"].rstrip().endswith("?") for l in mel),
          "of which followed by a voice change:", sum(1 for i in range(1, len(mel)) if mel[i - 1]["t"].rstrip().endswith("?") and i in vc))
