import subprocess, sys
from pathlib import Path
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path(sys.argv[1])
if not p.exists():
    sys.exit(f"Path does not exist: {p}")
files = sorted(p.rglob("*.wav"))
if not files:
    sys.exit(f"No .wav files found in: {p}")

def probe(f):
    try:
        info = sf.info(f)
        return info.frames / info.samplerate, info.samplerate, info.channels
    except Exception:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels:format=duration",
             "-of", "default=nw=1:nk=1", str(f)],
            capture_output=True, text=True).stdout.split()
        sr, ch, dur = int(out[0]), int(out[1]), float(out[2])
        return dur, sr, ch

durations, rates, channels, bad = [], [], [], []
for f in files:
    try:
        d_, sr_, ch_ = probe(f)
        durations.append(d_); rates.append(sr_); channels.append(ch_)
    except Exception:
        bad.append(f.name)

d = pd.Series(durations)
print("Files       :", len(files), f"({len(bad)} unreadable)")
print("Total hours :", round(d.sum() / 3600, 2))
print("Sample rates:\n", pd.Series(rates).value_counts().to_string())
print("Channels:\n", pd.Series(channels).value_counts().to_string())
print("Duration (sec):\n", d.describe().round(2).to_string())
if bad:
    print("Unreadable (first 10):", bad[:10])

plt.hist(d, bins=50)
plt.xlabel("duration (sec)")
plt.ylabel("count")
plt.title(f"{p.parent.name}  |  {d.sum()/3600:.2f} hours, {len(d)} samples")
plt.savefig(f"durations_{p.parent.name}.png", dpi=120)

# for running: 
# uv run python data_stats_mendeley.py "/home/dilmurod/job/tts_project/data/Karakalpak_Speech_Corpus_mendeley_v1/DATASET"