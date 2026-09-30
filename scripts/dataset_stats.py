import io, sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = Path(sys.argv[1])
files = sorted(p.rglob("*.parquet")) if p.is_dir() else [p]

durations, rates = [], []
for f in files:
    for a in pq.read_table(f, columns=["audio"]).column("audio").to_pylist():
        info = sf.info(io.BytesIO(a["bytes"]))
        durations.append(info.frames / info.samplerate)
        rates.append(info.samplerate)

d = pd.Series(durations)
print("Files       :", len(files))
print("Samples     :", len(d))
print("Total hours :", round(d.sum() / 3600, 2))
print("Sample rates:\n", pd.Series(rates).value_counts().to_string())
print("Duration (sec):\n", d.describe().round(2).to_string())

plt.hist(d, bins=50)
plt.xlabel("duration (sec)")
plt.ylabel("count")
plt.title(f"{p.name}  |  {d.sum()/3600:.2f} hours, {len(d)} samples")
plt.savefig(f"durations_{p.stem if p.is_file() else p.name}.png", dpi=120)

# to run: uv run python audio_stats.py path_to_folder_or_path_to_parquet_file