"""The Koppel episode transcribed the desktop's way (faster-whisper, word times) and given its
voices from the pyannote turns with the desktop's own `with_voices` (copied), so that it is a second
test set of the same kind as the Mel Robbins one.  transcribe_koppel.py wav turns.json out.json"""
import json, sys, time
from faster_whisper import WhisperModel

wav, turns_file, out = sys.argv[1:4]
model_name = sys.argv[4] if len(sys.argv) > 4 else "large-v3-turbo"
t0 = time.time()
model = WhisperModel(model_name, device="cuda", compute_type="float16")
segments, info = model.transcribe(wav, language="en", vad_filter=True, word_timestamps=True)
lines, words = [], []
for s in segments:
    lines.append((int(s.start * 1000), int(s.end * 1000), s.text))
    words.append([(int(w.start * 1000), int(w.end * 1000), w.word) for w in (s.words or [])])
print("whisper", model_name, len(lines), "lines in", time.time() - t0, "s")

turns = sorted((int(t["a"] * 1000), int(t["b"] * 1000), t["s"]) for t in json.load(open(turns_file))["turns"])


def _voice_over(turns, a, b):
    share = {}
    for ta, tb, voice in turns:
        if tb <= a:
            continue
        if ta >= b:
            break
        share[voice] = share.get(voice, 0) + min(b, tb) - max(a, ta)
    return max(share, key=share.get) if share else None


def with_voices(lines, words, turns):
    out, voices = [], []
    for i, (a, b, t) in enumerate(lines):
        ws = words[i] if words and i < len(words) else None
        if not ws or len(ws) < 4:
            out.append((a, b, t))
            voices.append(_voice_over(turns, a, b))
            continue
        said = [_voice_over(turns, wa, max(wb, wa + 1)) for wa, wb, _w in ws]
        for k in range(len(said)):
            if said[k] is None:
                said[k] = next((v for v in said[k + 1:] if v is not None), None) if k == 0 or said[k - 1] is None else said[k - 1]
        runs = []
        for k, v in enumerate(said):
            if runs and runs[-1][0] == v:
                runs[-1][1].append(k)
            else:
                runs.append([v, [k]])
        merged = []
        for v, ks in runs:
            if merged and (len(ks) < 3 or merged[-1][0] == v):
                merged[-1][1].extend(ks)
            else:
                merged.append([v, list(ks)])
        if len(merged) > 1 and len(merged[0][1]) < 3:
            merged[1][1] = merged[0][1] + merged[1][1]
            merged.pop(0)
        for v, ks in merged:
            text = "".join(ws[k][2] for k in ks).strip()
            if text:
                out.append((ws[ks[0]][0], ws[ks[-1]][1], text))
                voices.append(v)
    numbers = {}
    return out, [None if v is None else numbers.setdefault(v, len(numbers)) for v in voices]


rows, voices = with_voices(lines, words, turns)
data = [{"a": a, "b": b, "t": t.strip(), **({"s": v} if v is not None else {})} for (a, b, t), v in zip(rows, voices)]
json.dump({"language": "en", "lines": data}, open(out, "w"), ensure_ascii=False)
print("saved", out, len(data), "lines")
