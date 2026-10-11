"""pyannote community-1 on a 16 kHz wav -> turns JSON (same pipeline as the desktop app)."""
import array, json, sys, time, wave
import torch
from pyannote.audio import Pipeline

wav_path, out_path = sys.argv[1], sys.argv[2]
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
with wave.open(wav_path, "rb") as w:
    rate, frames = w.getframerate(), w.readframes(w.getnframes())
samples = torch.tensor(array.array("h", frames), dtype=torch.float32).unsqueeze(0) / 32768.0
t = time.time()
out = pipe({"waveform": samples, "sample_rate": rate})
ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", None) or out
turns = [{"a": turn.start, "b": turn.end, "s": str(speaker)} for turn, _t, speaker in ann.itertracks(yield_label=True)]
json.dump({"seconds": time.time() - t, "audio_seconds": samples.shape[1] / rate, "turns": turns}, open(out_path, "w"))
print("done", len(turns), "turns in", time.time() - t, "s")
