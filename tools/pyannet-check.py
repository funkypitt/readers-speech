#!/usr/bin/env python
"""Compares pyannet.c with onnxruntime on the same windows, on the host.

    pyannet-check.py model.onnx model.bin check-binary wav [wav ...] [--windows N] [--frames out.json]

For each wav: the whole file when it is 30 s or shorter (the fixture: the five windows the
detector looks at, plus the full excerpt); otherwise N ten-second windows spread over it.
Prints, per window, the max |difference| of the log-probabilities, the share of frames whose
likeliest class agrees, and the C time on one core. With --frames, writes the fixture's windows
(offset, length, class per frame, from onnxruntime) for VoicesTest.
Run with the interview env: ~/miniconda3/envs/interview/bin/python."""
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import onnxruntime as ort

SR = 16000
WINDOW = 10 * SR
MARGIN = 2 * SR

opts, args = {}, []
it = iter(sys.argv[1:])
for a in it:
    if a.startswith("--"):
        opts[a] = next(it)
    else:
        args.append(a)
onnx_path, bin_path, check, wavs = args[0], args[1], args[2], args[3:]
n_windows = int(opts.get("--windows", 6))
frames_out = opts.get("--frames")

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.log_severity_level = 3
sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def fixture_windows(pcm, fixture):
    """The windows Voices.mark looks at on the fixture (gated gaps), as (offset, n)."""
    out = []
    for g in fixture["expected_gate"]["gaps"]:
        gap = g["gap_sample"]
        a = max(0, gap - WINDOW // 2)
        b = min(len(pcm), a + WINDOW)
        out.append((a, b - a))
    return out


worst_diff, worst_agree, times = 0.0, 1.0, []
recorded = []
tmp = tempfile.mkdtemp(prefix="pyannet-")
for wav in wavs:
    pcm = read_wav(wav)
    windows = []
    fixture = None
    fj = os.path.splitext(wav)[0] + ".json"
    if len(pcm) <= 30 * SR:
        if os.path.isfile(fj):
            fixture = json.load(open(fj))
            windows += fixture_windows(pcm, fixture)
        windows.append((0, len(pcm)))
    else:
        step = (len(pcm) - WINDOW) // (n_windows + 1)
        windows = [(step * (i + 1), WINDOW) for i in range(n_windows)]
    print(os.path.basename(wav), "%.0f s, %d windows" % (len(pcm) / SR, len(windows)))
    for off, n in windows:
        y_ref = sess.run(None, {name: pcm[None, None, off:off + n]})[0][0]
        out = os.path.join(tmp, "out.f32")
        res = subprocess.run([check, bin_path, wav, str(off), str(n), out], check=True, capture_output=True, text=True)
        frames, ms = res.stdout.split()
        frames, ms = int(frames), float(ms)
        y = np.fromfile(out, np.float32).reshape(frames, 7)
        assert y.shape == y_ref.shape, (y.shape, y_ref.shape)
        diff = float(np.abs(y - y_ref).max())
        agree = float((y.argmax(1) == y_ref.argmax(1)).mean())
        prob_diff = float(np.abs(np.exp(y) - np.exp(y_ref)).max())
        worst_diff, worst_agree = max(worst_diff, diff), min(worst_agree, agree)
        if n == WINDOW:
            times.append(ms)
        print("  offset %8d  n %6d  frames %3d  max|dlogp| %.2e  max|dp| %.2e  argmax agree %.1f%%  C %.1f ms" % (off, n, frames, diff, prob_diff, 100 * agree, ms))
        if fixture is not None and (off, n) != (0, len(pcm)):
            recorded.append(dict(offset=off, n=n, classes=[int(c) for c in y_ref.argmax(1)]))
print("worst: max|dlogp| %.2e, argmax agreement %.1f%%; C time per 10 s window: median %.1f ms, max %.1f ms (one core)" % (
    worst_diff, 100 * worst_agree, float(np.median(times)) if times else float("nan"), max(times) if times else float("nan")))
if frames_out:
    json.dump(dict(description="per window the detector runs on fixture.wav (offset and length in samples), the likeliest class per frame"
                               " as onnxruntime computes it on the model; classes: none, A, B, C, A+B, A+C, B+C",
                   windows=recorded), open(frames_out, "w"), separators=(",", ":"))
    print("saved", frames_out, len(recorded), "windows")
sys.exit(0 if worst_diff <= 1e-3 and worst_agree == 1.0 else 1)
