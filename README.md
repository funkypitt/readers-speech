# readers-speech

Speech on the phone for the Reader's apps — [Reader's Audio Player](https://github.com/funkypitt/readers-audio)
and [Reader's Recorder](https://github.com/funkypitt/readers-recorder): transcription with a vendored
whisper.cpp, the main points of a transcript with a vendored llama.cpp, and one copy of each model
shared between the two apps.

An Android library module, consumed as a git submodule at `speech/` in each app:

    git clone --recursive https://github.com/funkypitt/readers-audio.git
    # or, in an existing checkout:
    git submodule update --init

`settings.gradle.kts` includes it as `:speech`; the app depends on `project(":speech")`.

## What is in it

- `src/main/cpp` — whisper.cpp 1.9.4 and llama.cpp, compiled straight into one shared library per
  CPU flavour (arm64 baseline and arm64 with fp16 arithmetic; x86_64 for the emulator), plus the two
  JNI bridges `jni.c` and `llama_jni.c`.
- `whisper/` — `WhisperLib`, `WhisperSession`, the two model qualities (`Models`), the punctuation
  prompts and the paragraph rule (`Prompts`, `Paragraphs`).
- `audio/` — MediaCodec decoding in pieces and the resampler to 16 kHz.
- `summary/` — `Summariser` (notes per part, then the theme, then the points of the whole talk),
  `SummaryModel` (Qwen2.5-3B-Instruct, fetched once, resumable), `LlamaLib`.
- `share/` — `ModelProvider` and `ModelFiles`: a model downloaded by one app is opened by the other
  through a content provider, by file descriptor; nothing is copied.

MIT, like the apps.
