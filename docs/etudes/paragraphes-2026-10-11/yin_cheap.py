"""The cheap pitch: plain YIN (no HMM) on the whole file, timed, then the per-line mean log-F0 over
the frames that are loud and periodic enough. Writes <feat>.yin.npz with f0_yin per line, and the
time it took.  yin_cheap.py transcript.json audio.wav feat.npz"""
import json, sys, time
import numpy as np
import soundfile as sf
import librosa

tr, wav, feat = sys.argv[1:4]
lines = json.load(open(tr))["lines"]
y, sr = sf.read(wav, dtype="float32")
HOP = 160
t0 = time.time()
f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=HOP, trough_threshold=0.1)
t_yin = time.time() - t0
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
# voicing proxy without pyin: a frame counts when it is loud for its line and its YIN estimate is
# not pinned at the search bounds (what YIN returns for noise)
out = []
for l in lines:
    fa, fb = int(l["a"] * sr / 1000) // HOP, max(int(l["b"] * sr / 1000) // HOP, int(l["a"] * sr / 1000) // HOP + 5)
    r, f = rms[fa:fb], f0[fa:fb]
    n = min(len(r), len(f))
    r, f = r[:n], f[:n]
    keep = (r > max(r.max() * 0.2, 1e-4)) & (f > 65) & (f < 390) if n else np.zeros(0, bool)
    out.append(float(np.mean(np.log(f[keep]))) if keep.sum() > 5 else np.nan)
F = np.load(feat)
a, b = np.array(out), F["f0"]
ok = ~np.isnan(a) & ~np.isnan(b)
print("yin whole file: %.1f s = %.4f x real time; corr with pyin per-line mean log-F0: %.3f (n=%d)" % (
    t_yin, t_yin / (len(y) / sr), np.corrcoef(a[ok], b[ok])[0, 1], ok.sum()))
np.savez(feat.replace(".npz", ".yin.npz"), f0_yin=a, t_yin=t_yin, audio_s=len(y) / sr)
