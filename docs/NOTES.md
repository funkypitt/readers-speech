# readers-speech — notes

Moved out of the README: what the module holds, and what was measured.

## What is in it

- `src/main/cpp` — whisper.cpp 1.9.4 and llama.cpp, compiled straight into one shared library per
  CPU flavour (arm64 baseline and arm64 with fp16 arithmetic; x86_64 for the emulator), plus the two
  JNI bridges `jni.c` and `llama_jni.c`, and `pyannet.c`, pyannote's segmentation network written
  out in plain C (in the whisper library; see below).
- `whisper/` — `WhisperLib`, `WhisperSession`, the two model qualities (`Models`), the punctuation
  prompts and the paragraph rule (`Prompts`, `Paragraphs`), `Vad`, the Silero voice detector
  carried in the assets so that silence is never decoded, and `Voices`, pyannote's segmentation
  model (its weights in the assets, the network in `pyannet.c`) that marks the lines where
  another person starts speaking, so that `Paragraphs` cuts there. `Segment.newVoice` carries
  the mark; a `WhisperSession` given a `Voices` sets it while the piece of sound is still in
  memory (gaps between two pieces are blind).

## The segmentation network in C, not an engine

The first version of `Voices` (2026-10-11, morning) ran the model with ONNX Runtime; its
`onnxruntime-android` brought 34–41 MB of native library per ABI, and Reader's Audio Player went
from 46 to 127 MB. The same afternoon the network was written out by hand in `src/main/cpp/pyannet.c`
(plain C99, libm only, single-threaded, about 450 lines): the graph read node by node from the
ONNX file — an instance normalisation of the raw waveform first, then the SincNet front end
(the sinc filters are baked as constants: conv 1→80 kernel 251 stride 10, abs, max-pool 3,
instance norm, leaky relu 0.01; two more conv-pool-norm-relu blocks 80→60 and 60→60, kernel 5),
four bidirectional LSTM layers of 128 (gates in ONNX order i, o, f, c; sigmoid/tanh), two linear
layers of 128 with leaky relu, a linear to seven classes and a log-softmax. The weights are the
ONNX initialisers, written as float32 by `tools/export-pyannet.py` into
`src/main/assets/pyannote-segmentation-3.0.bin` (layout documented in that script).

Checked on the host, not only on the phone: `tools/pyannet-check.sh` compiles `pyannet.c` with gcc and
compares it with onnxruntime on the test fixture and on ten-second windows of the two episodes of
the study — 22 windows, max |Δ log p| 7.8e-5 (3.1e-4 on a thirty-second window), every frame's
likeliest class the same. One desktop core: 50–60 ms per ten-second window with gcc or the NDK's
clang (onnxruntime: 33 ms). The heavy loops are one register-tiled kernel (four rows × sixteen
columns) that the compiler vectorises without reassociating floating point, so the numbers are
the same on every CPU. `VoicesTest` on the JVM cannot reach the C: it tests the Kotlin layer on
the frames onnxruntime computed for the fixture's windows (`fixture-frames.json`).
- `translate/` — `Translator` and `TranslateModel`: a transcript rendered into another language on
  the phone, in blocks that keep their place in the sound.
- `audio/` — MediaCodec decoding in pieces and the resampler to 16 kHz.
- `summary/` — `Summariser` (notes per part, then the theme, then the points of the whole talk),
  `SummaryModel` (Qwen2.5-3B-Instruct, fetched once, resumable), `LlamaLib`.
- `share/` — `ModelProvider` and `ModelFiles`: a model downloaded by one app is opened by the other
  through a content provider, by file descriptor; nothing is copied.

## Two things measured rather than assumed

**Which model.** The careful one (large-v3-turbo) gets about twice as many words right as the
ordinary one (small) — 4 to 5 % wrong against 10 to 12 % — and takes four times as long. On a
telephone that is the difference between half an hour and an afternoon for an hour-long talk, so
the ordinary model is what the apps advise where the recordings are long (Podcasts, Audio Player)
and the careful one where they are short notes (Recorder).

**The voice detector.** Measured on 2026-09-18 against a reference that is not another
transcription — a LibriVox reading of Daudet scored word by word against Gutenberg's own text.
On three minutes of continuous reading it changes nothing either way; on the same three minutes
with pauses it saves a tenth of the time and, with the ordinary model, two points of word error
(12.1 → 10.0 %). The padding is the whole safety of it: at 30 ms the detector eats the first words
of a sentence (12.8 %), at 250 ms it does not. Enabled in all three apps.

One vendor patch came out of that work: ggml's `llamafile_sgemm` asserted on the strides the
detector hands it, and the apps compile ggml with `GGML_USE_LLAMAFILE` where whisper.cpp's own
build does not — hence a detector that measured well on the desktop and killed the app on Android.
The kernel now declines such a request, which both callers already handle. See the comment in
`sgemm.cpp`, and `src/androidTest` for the check that found it.
