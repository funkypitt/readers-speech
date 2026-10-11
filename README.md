# readers-speech

Speech on the phone for the Reader's apps: transcription (whisper.cpp), the main points of a
transcript and translation (llama.cpp), and one copy of each model shared between the apps.
An Android library module, used as a git submodule by
[Reader's Recorder](https://github.com/funkypitt/readers-recorder),
[Audio Player](https://github.com/funkypitt/readers-audio),
[Podcasts](https://github.com/funkypitt/readers-podcasts) and
[Notes](https://github.com/funkypitt/readers-notes). Nothing is uploaded.

## Key points

* Two transcription qualities: normal (Whisper small, 190 MB) and high (large-v3-turbo, 574 MB):
  about half the word errors (4–5 % against 10–12 %), four times as long.
* A Silero voice detector, carried in the assets, keeps silence from being decoded (padding
  250 ms). Audio is decoded in pieces, so memory stays flat whatever the length.
* Paragraphs where the speech breathes and where the voice changes: pyannote's segmentation
  model alone (six megabytes of weights, MIT, in the assets, run by a C implementation of the
  network in the module — no inference engine) looks at ten seconds around each gap between two
  lines and says whether another person speaks after it; a question and its answer become two
  paragraphs, with no speaker labels. The same paragraph rule as the desktop apps (pauses of
  2 s, 0.9 s after a sentence, long paragraphs). About a minute of CPU per hour of sound;
  without the model, the pauses decide alone (`docs/etudes/paragraphes-2026-10-11`).
* Main points: Qwen2.5-3B-Instruct (about 2 GB) — notes per part, then the theme, then the
  points. Refused on a phone with too little memory.
* Translation: Gemma 3 4B (about 2.5 GB), in blocks that keep their place in the sound; needs
  an 8 GB phone.
* Models are fetched once, resumable, and shared: an app opens a model another app downloaded,
  through a content provider, by file descriptor. Only between apps signed with the same key.
* whisper.cpp 1.9.4 and llama.cpp are vendored and compiled per CPU flavour (arm64 baseline,
  fp16, dotprod+i8mm; x86_64 for the emulator).
* One vendor patch, in ggml's `sgemm.cpp`: without it the voice detector kills the app on
  Android only. Keep it when updating the vendored code.

More detail: [docs/NOTES.md](docs/NOTES.md) — the layout, the measurements, the patch.

## Use

    git clone --recursive https://github.com/funkypitt/readers-audio.git
    # or, in an existing checkout:
    git submodule update --init

The submodule sits at `speech/`; `settings.gradle.kts` includes it as `:speech`, the app depends
on `project(":speech")`.

## Build

Built with the app (NDK 27.1.12297006, CMake). The app must name the same `ndkVersion` in its own
`build.gradle.kts`, or the native libraries ship unstripped. `src/androidTest` holds the
on-device check of the native path.

MIT, like the apps.
