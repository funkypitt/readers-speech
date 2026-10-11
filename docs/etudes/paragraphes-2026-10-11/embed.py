"""Candidate 4: per-line speaker embeddings with the small models inside pyannote community-1,
on CPU with one thread, timed, so that the cost on a phone can be estimated.
 - WeSpeakerResNet34 (6.6 M params) embedding of each line's span (min 1.5 s, max 6 s).
 - PyanNet segmentation (1.5 M params) on a 10 s window centred on each gap between lines:
   the local speaker active in the 2 s before the gap vs the 2 s after.
Usage: embed.py transcript.json audio.wav out.npz [device]"""
import json, sys, time
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline, Inference

torch.set_num_threads(1)
tr, wav, out = sys.argv[1:4]
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
lines = json.load(open(tr))["lines"]
y, sr = sf.read(wav, dtype="float32")
assert sr == 16000

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
emb_model = pipe._embedding.model_ if hasattr(pipe._embedding, "model_") else pipe._embedding.model
seg_model = pipe._segmentation.model
emb_model.to(device).eval()
seg_model.to(device).eval()
print("emb", type(emb_model).__name__, sum(p.numel() for p in emb_model.parameters()) / 1e6, "M")
print("seg", type(seg_model).__name__, sum(p.numel() for p in seg_model.parameters()) / 1e6, "M")
print("seg specs", seg_model.specifications)

def chunk(a_ms, b_ms, min_s=1.5, max_s=6.0):
    a, b = int(a_ms * sr / 1000), int(b_ms * sr / 1000)
    if b - a < min_s * sr:
        c = (a + b) // 2
        a, b = max(0, c - int(min_s * sr / 2)), min(len(y), c + int(min_s * sr / 2))
    if b - a > max_s * sr:
        b = a + int(max_s * sr)
    return torch.from_numpy(y[a:b]).unsqueeze(0).unsqueeze(0)

# ---- embeddings ----
embs, secs = [], 0.0
t0 = time.time()
with torch.inference_mode():
    for i, l in enumerate(lines):
        x = chunk(l["a"], l["b"]).to(device)
        secs += x.shape[-1] / sr
        e = emb_model(x)
        embs.append(e.squeeze(0).cpu().numpy())
        if i % 200 == 0:
            print("emb", i, "%.1f s" % (time.time() - t0))
t_emb = time.time() - t0
print("embedding: %.1f s for %.0f s of audio -> %.3f x real time (1 thread, %s)" % (t_emb, secs, t_emb / secs, device))

# ---- local segmentation around each gap ----
inf = Inference(seg_model, window="whole", device=torch.device(device))
WIN, PRE = 10.0, 2.0
before, after, changes, secs2 = [], [], [], 0.0
t0 = time.time()
for i, l in enumerate(lines):
    if i == 0:
        before.append(-1); after.append(-1); changes.append(0.0)
        continue
    gap = (lines[i - 1]["b"] + l["a"]) / 2000.0
    a = max(0.0, gap - WIN / 2)
    b = min(len(y) / sr, a + WIN)
    x = torch.from_numpy(y[int(a * sr):int(b * sr)]).unsqueeze(0)
    secs2 += x.shape[-1] / sr
    with torch.inference_mode():
        act = inf({"waveform": x, "sample_rate": sr})          # SlidingWindowFeature (frames, speakers)
    data = act if isinstance(act, np.ndarray) else act.data           # (frames, local speakers), multilabel
    step = (b - a) / len(data)
    def frames(t1, t2):
        f1, f2 = int(max(0, (t1 - a) / step)), int(min(len(data), (t2 - a) / step))
        return data[f1:max(f2, f1 + 1)]
    pre, post = frames(gap - PRE, gap).sum(0), frames(gap, gap + PRE).sum(0)
    sb = int(pre.argmax()) if pre.max() > 0 else -1
    sa = int(post.argmax()) if post.max() > 0 else -1
    before.append(sb); after.append(sa)
    # a continuous score: share of the post window held by the pre speaker (low = change)
    changes.append(float(1.0 - (post[sb] / post.sum() if sb >= 0 and post.sum() > 0 else 1.0)))
    if i % 200 == 0:
        print("seg", i, "%.1f s" % (time.time() - t0))
t_seg = time.time() - t0
print("segmentation: %.1f s for %.0f s of windows -> %.3f x window time, %.3f x audio time (1 thread, %s)" %
      (t_seg, secs2, t_seg / secs2, t_seg / (len(y) / sr), device))
np.savez(out, emb=np.stack(embs), seg_before=np.array(before), seg_after=np.array(after), seg_change=np.array(changes),
         t_emb=t_emb, emb_secs=secs, t_seg=t_seg, seg_secs=secs2, audio_s=len(y) / sr)
print("saved", out)
