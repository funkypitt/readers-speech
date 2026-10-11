"""Cheap per-line acoustic features (candidate 3): mean/std of 13 MFCCs (whisper-like 25 ms / 10 ms
frames, 80 mels) over the voiced frames of a line's span, and the mean log-F0 (pyin) over its voiced
frames. Usage: features.py transcript.json audio.wav out.npz"""
import json, sys, time
import numpy as np
import soundfile as sf
import librosa

tr, wav, out = sys.argv[1:4]
lines = json.load(open(tr))["lines"]
y, sr = sf.read(wav, dtype="float32")
assert sr == 16000
print("audio", len(y) / sr, "s")

t0 = time.time()
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=80)
logmel = np.log10(np.maximum(mel, 1e-10))
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)          # (13, frames)
rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
t_mfcc = time.time() - t0
print("mfcc whole file: %.1f s (%.4f x real time)" % (t_mfcc, t_mfcc / (len(y) / sr)))

HOP = 160
feats = []
t0 = time.time()
for i, l in enumerate(lines):
    a, b = int(l["a"] * sr / 1000), int(l["b"] * sr / 1000)
    fa, fb = a // HOP, max(b // HOP, a // HOP + 5)
    seg_rms = rms[fa:fb]
    thr = max(seg_rms.max() * 0.1, 1e-4) if len(seg_rms) else 1e-4
    keep = seg_rms > thr
    m = mfcc[:, fa:fb][:, keep] if keep.any() else mfcc[:, fa:fb]
    lm = logmel[:, fa:fb][:, keep] if keep.any() else logmel[:, fa:fb]
    seg = y[a:b]
    f0 = np.nan
    if len(seg) > sr // 4:
        f, voiced, _p = librosa.pyin(seg, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=160)
        if voiced.sum() > 5:
            f0 = float(np.nanmean(np.log(f[voiced])))
            f0std = float(np.nanstd(np.log(f[voiced])))
        else:
            f0std = np.nan
    else:
        f0std = np.nan
    feats.append(dict(mfcc_mean=m.mean(1), mfcc_std=m.std(1), logmel_mean=lm.mean(1), f0=f0, f0std=f0std,
                      voiced_frames=int(keep.sum()), rms=float(seg_rms.mean() if len(seg_rms) else 0)))
    if i % 200 == 0:
        print(i, time.time() - t0)
print("per-line pyin+means: %.1f s" % (time.time() - t0))
np.savez(out, mfcc_mean=np.stack([f["mfcc_mean"] for f in feats]), mfcc_std=np.stack([f["mfcc_std"] for f in feats]),
         logmel_mean=np.stack([f["logmel_mean"] for f in feats]), f0=np.array([f["f0"] for f in feats]),
         f0std=np.array([f["f0std"] for f in feats]), voiced=np.array([f["voiced_frames"] for f in feats]),
         rms=np.array([f["rms"] for f in feats]), t_mfcc=t_mfcc, audio_s=len(y) / sr)
print("saved", out)
