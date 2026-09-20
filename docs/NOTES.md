# readers-speech — notes

Moved out of the README: what the module holds, and what was measured.

## What is in it

- `src/main/cpp` — whisper.cpp 1.9.4 and llama.cpp, compiled straight into one shared library per
  CPU flavour (arm64 baseline and arm64 with fp16 arithmetic; x86_64 for the emulator), plus the two
  JNI bridges `jni.c` and `llama_jni.c`.
- `whisper/` — `WhisperLib`, `WhisperSession`, the two model qualities (`Models`), the punctuation
  prompts and the paragraph rule (`Prompts`, `Paragraphs`), and `Vad`, the Silero voice detector
  carried in the assets so that silence is never decoded.
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
