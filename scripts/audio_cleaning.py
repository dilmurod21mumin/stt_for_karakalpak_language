"""Trim leading/trailing silence from a Parquet speech dataset.

decode -> mono -> resample to 16 kHz (torchaudio) -> Silero VAD -> keep first_start:last_end
Output: <project>/data/cleaned/<name>/parquet_files/<same file name>  (script lives in <project>/scripts/)  with columns  id | cleaned_audio | raw_text

Usage: python audio_cleaning.py <file.parquet | folder> [--name my_dataset] --workers number_of_cpu_cores
"""

"""
Parquet
   ↓
read audio bytes
   ↓
decode audio with soundfile
   ↓
convert stereo → mono
   ↓
resample → 16 kHz
   ↓
Silero VAD
   ↓
find first speech start + last speech end
   ↓
trim only outer silence
   ↓
encode as OGG/Opus
   ↓
write cleaned Parquet
"""


import argparse, io, os, re
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps
from tqdm import tqdm

SR = 16_000
vad = None
resamplers = {}

AUDIO = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def init_worker():
    """Each worker process: 1 CPU thread + its own Silero model (CPU)."""
    global vad
    torch.set_num_threads(1)
    vad = load_silero_vad()


def clean(audio_bytes):
    """bytes -> (cleaned ogg bytes) or None if no speech."""
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    wav = torch.from_numpy(wav).float()
    wav = wav.unsqueeze(0) if wav.ndim == 1 else wav.mean(1, keepdim=True).T  # mono [1, n]
    if sr != SR:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, SR)
        wav = resamplers[sr](wav)
    wav = wav.squeeze(0)

    ts = get_speech_timestamps(wav, vad, sampling_rate=SR, threshold=0.5,
                               min_speech_duration_ms=250, min_silence_duration_ms=100,
                               speech_pad_ms=30)
    if not ts:
        return None
    wav = wav[ts[0]["start"]: ts[-1]["end"]]  # everything in between is kept as is

    buf = io.BytesIO()
    sf.write(buf, wav.numpy(), SR, format="OGG", subtype="OPUS")
    return buf.getvalue()


def work(item):
    sid, a = item
    try:
        return sid, clean(a["bytes"]), None
    except Exception as e:
        return sid, None, f"error: {e}"


def process(src, dst, pool):
    pf = pq.ParquetFile(src)
    schema = pa.schema([pf.schema_arrow.field("id"), ("cleaned_audio", AUDIO),
                        pf.schema_arrow.field("raw_text")])
    skipped = []
    with pq.ParquetWriter(dst, schema) as writer, tqdm(total=pf.metadata.num_rows, desc=src.name) as bar:
        for batch in pf.iter_batches(batch_size=256, columns=["id", "audio", "raw_text"]):
            ids = batch["id"].to_pylist()
            results = pool.imap(work, zip(ids, batch["audio"].to_pylist()), chunksize=4)  # keeps order
            keep, audio = [], []
            for i, (sid, out, err) in enumerate(results):
                bar.update(1)
                if out is None:
                    skipped.append((sid, err or "no speech"))
                    continue
                keep.append(i)
                audio.append({"bytes": out, "path": f"{sid}.ogg"})
            if keep:
                idx = pa.array(keep)
                writer.write_table(pa.table({
                    "id": batch["id"].take(idx),
                    "cleaned_audio": pa.array(audio, type=AUDIO),
                    "raw_text": batch["raw_text"].take(idx),   # untouched
                }, schema=schema))
    print(f"{src.name}: skipped {len(skipped)}", *skipped[:10], sep="\n  ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.parquet"))
    folder = args.input.parent if args.input.is_file() else args.input
    name = args.name or re.sub(r"\W+", "_", folder.resolve().name).strip("_").lower()
    out_dir = Path(__file__).resolve().parent.parent / "data" / "cleaned" / name / "parquet_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=init_worker) as pool:
        for f in files:
            process(f, out_dir / f.name, pool)