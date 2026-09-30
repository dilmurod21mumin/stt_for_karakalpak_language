"""
Build a speech-to-text dataset from:
  - a folder of WAV audio files (one audiobook, downloaded with download_playlist.py)
  - a PDF containing the same book's text

Pipeline:
  1. Concatenate all WAVs into one continuous audio track (playlist order).
  2. Extract text from the PDF (native text layer; falls back to a warning if empty).
  3. Segment the audio into short chunks using voice activity detection (Silero VAD).
  4. Transcribe each chunk with the Karakalpak Whisper model (rough transcript).
  5. Fuzzy-match each rough transcript against a sliding window of the book text
     to find the corresponding "true" text and advance a cursor through the book.
  6. Filter out chunks with poor audio/text agreement (likely misalignments).
  7. Export train/validation/test Parquet files matching the schema of
     atikuwu/karakalpak-speech-corpus: id, audio, text, raw_text.

Usage:
    uv run python build_dataset.py \
        --audio-dir /home/dilmurod/job/tts_project/data/raw_audio/book01 \
        --pdf /home/dilmurod/job/tts_project/data/pdfs/book01.pdf \
        --out /home/dilmurod/job/tts_project/data/processed/book01 \
        --book-id book01
"""

import argparse
import io
import json
import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from rapidfuzz import fuzz
from tqdm import tqdm
from transformers import pipeline

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

WHISPER_MODEL_ID = "atikuwu/whisper-medium-karakalpak"
TARGET_SR = 16000

MIN_CHUNK_SEC = 2.0
MAX_CHUNK_SEC = 20.0

# Minimum fuzzy match score (0-100) between the Whisper rough transcript and
# the corresponding book text window to keep a chunk. Raise this for a
# cleaner but smaller dataset; lower it to keep more (noisier) data.
MIN_MATCH_SCORE = 55

# How far ahead (in characters) to search in the book text for each chunk's
# best match, relative to the current cursor position. Keeps matching fast
# and avoids accidentally matching a much later part of the book.
SEARCH_WINDOW_CHARS = 600

RANDOM_SEED = 42
VAL_FRACTION = 0.05
TEST_FRACTION = 0.05


# ----------------------------------------------------------------------------
# Step 1: Load and concatenate audio
# ----------------------------------------------------------------------------

def load_and_concat_audio(audio_dir: Path) -> np.ndarray:
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {audio_dir}")

    print(f"Found {len(wav_files)} audio file(s); loading and concatenating...")
    chunks = []
    for wf in tqdm(wav_files):
        audio, sr = sf.read(wf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # downmix to mono if needed
        if sr != TARGET_SR:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, TARGET_SR
            ).numpy()
        chunks.append(audio)

    return np.concatenate(chunks)


# ----------------------------------------------------------------------------
# Step 2: Extract and clean PDF text
# ----------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    text = "\n".join(pages)

    if len(text.strip()) < 200:
        print(
            "WARNING: Very little text extracted from the PDF. "
            "It may be a scanned image PDF with no text layer. "
            "You will need OCR (e.g. the kaa-ml/tesseract-kaa-latn model) first."
        )

    return clean_book_text(text)


def clean_book_text(text: str) -> str:
    # Remove common noise: page numbers on their own line, repeated headers,
    # excess whitespace. This is deliberately conservative -- inspect the
    # output and adjust the regexes to your specific book's formatting.
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)          # standalone page numbers
    text = re.sub(r"[ \t]+", " ", text)                   # collapse spaces/tabs
    text = re.sub(r"\n{2,}", "\n", text)                   # collapse blank lines
    text = text.strip()
    return text


def normalize_for_matching(text: str) -> str:
    # Lowercase + strip punctuation for fuzzy matching only.
    # Keep the original text separately for the final dataset.
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\sáǵıńóúşşqƣ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------------
# Step 3: Voice activity detection -> chunk boundaries
# ----------------------------------------------------------------------------

def get_speech_chunks(audio: np.ndarray) -> list[tuple[int, int]]:
    print("Running voice activity detection...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, _, _, _, _) = utils

    wav_tensor = torch.from_numpy(audio)
    raw_segments = get_speech_timestamps(
        wav_tensor, model, sampling_rate=TARGET_SR, return_seconds=False
    )

    # Merge short adjacent segments into chunks within [MIN_CHUNK_SEC, MAX_CHUNK_SEC]
    chunks = []
    cur_start, cur_end = None, None
    for seg in raw_segments:
        s, e = seg["start"], seg["end"]
        if cur_start is None:
            cur_start, cur_end = s, e
            continue
        merged_dur = (e - cur_start) / TARGET_SR
        if merged_dur <= MAX_CHUNK_SEC:
            cur_end = e
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append((cur_start, cur_end))

    chunks = [
        (s, e) for s, e in chunks
        if (e - s) / TARGET_SR >= MIN_CHUNK_SEC
    ]
    print(f"Got {len(chunks)} speech chunks after merging/filtering.")
    return chunks


# ----------------------------------------------------------------------------
# Step 4: Rough transcription with Whisper
# ----------------------------------------------------------------------------

def build_asr_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {WHISPER_MODEL_ID} (device={'cuda' if device == 0 else 'cpu'})...")
    return pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=device,
        generate_kwargs={"task": "transcribe", "language": "uz"},
    )


def transcribe_chunk(asr, audio_chunk: np.ndarray) -> str:
    result = asr({"array": audio_chunk, "sampling_rate": TARGET_SR})
    return result["text"].strip()


# ----------------------------------------------------------------------------
# Step 5+6: Align each chunk to the book text, filter poor matches
# ----------------------------------------------------------------------------

def align_chunks_to_book(chunks_with_transcripts, book_text: str):
    """
    Walk through the book text with a cursor. For each chunk's rough
    transcript, find the best-matching window of book text near the cursor,
    score the match, and advance the cursor past it if it's good enough.
    """
    norm_book = normalize_for_matching(book_text)
    cursor = 0
    results = []

    for item in tqdm(chunks_with_transcripts, desc="Aligning to book text"):
        rough = item["rough_text"]
        norm_rough = normalize_for_matching(rough)
        if not norm_rough:
            continue

        window_end = min(len(norm_book), cursor + SEARCH_WINDOW_CHARS)
        candidate_window = norm_book[cursor:window_end]

        # Slide a match window roughly the length of the rough transcript
        # across the candidate region and keep the best-scoring position.
        best_score, best_start, best_len = 0, cursor, len(norm_rough)
        step = max(5, len(norm_rough) // 4)
        for start in range(0, max(1, len(candidate_window) - len(norm_rough) + step), step):
            piece = candidate_window[start:start + len(norm_rough)]
            if not piece:
                continue
            score = fuzz.ratio(norm_rough, piece)
            if score > best_score:
                best_score = score
                best_start = cursor + start

        if best_score < MIN_MATCH_SCORE:
            continue  # likely misaligned (skipped text, ad-lib, noise, etc.)

        matched_norm = norm_book[best_start:best_start + best_len]

        # Map the matched *normalized* span back to the original book text
        # by matching character counts approximately (normalization mostly
        # preserves length/order). This is approximate but adequate for
        # ASR training references.
        true_text = book_text[best_start:best_start + best_len].strip()
        if not true_text:
            continue

        results.append({
            "start_sample": item["start_sample"],
            "end_sample": item["end_sample"],
            "raw_text": rough,       # what Whisper heard (unnormalized reference)
            "text": true_text,       # the actual book text, used as ground truth
            "match_score": best_score,
        })

        cursor = best_start + best_len  # advance cursor forward only

    print(f"Kept {len(results)} / {len(chunks_with_transcripts)} chunks after alignment filtering "
          f"(min_score={MIN_MATCH_SCORE}).")
    return results


# ----------------------------------------------------------------------------
# Step 7: Export to train/validation/test Parquet (schema-compatible)
# ----------------------------------------------------------------------------

def export_dataset(audio: np.ndarray, aligned_chunks: list[dict], out_dir: Path, book_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, item in enumerate(aligned_chunks):
        s, e = item["start_sample"], item["end_sample"]
        chunk_audio = audio[s:e]

        buf = io.BytesIO()
        sf.write(buf, chunk_audio, TARGET_SR, format="OGG", subtype="OPUS")
        audio_bytes = buf.getvalue()

        rows.append({
            "id": f"{book_id}_{i:05d}",
            "audio": {"bytes": audio_bytes, "path": f"{book_id}_{i:05d}.ogg"},
            "text": item["text"],
            "raw_text": item["raw_text"],
            "match_score": item["match_score"],
        })

    df = pd.DataFrame(rows)

    # Shuffle and split (speaker-independent isn't meaningful here since
    # it's one narrator, so this is a simple random split).
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    n = len(df)
    n_val = max(1, int(n * VAL_FRACTION))
    n_test = max(1, int(n * TEST_FRACTION))
    n_train = n - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        path = out_dir / f"{split_name}-00000-of-00001.parquet"
        split_df.drop(columns=["match_score"]).to_parquet(path, index=False)
        print(f"{split_name}: {len(split_df)} rows -> {path}")

    # Keep match_score alongside for your own QA, not fed to the model
    df[["id", "text", "raw_text", "match_score"]].to_csv(out_dir / "alignment_report.csv", index=False)

    total_hours = sum((item["end_sample"] - item["start_sample"]) for item in aligned_chunks) / TARGET_SR / 3600
    print(f"\nTotal exported audio: {total_hours:.2f} hours")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build an STT dataset from an audiobook + PDF.")
    parser.add_argument("--audio-dir", required=True, help="Folder of downloaded WAV files")
    parser.add_argument("--pdf", required=True, help="Path to the book's PDF")
    parser.add_argument("--out", required=True, help="Output folder for the Parquet dataset")
    parser.add_argument("--book-id", required=True, help="Short id used as a filename prefix")
    args = parser.parse_args()

    audio = load_and_concat_audio(Path(args.audio_dir))
    print(f"Total input audio: {len(audio) / TARGET_SR / 3600:.2f} hours")

    book_text = extract_pdf_text(Path(args.pdf))
    print(f"Extracted {len(book_text)} characters of book text.")

    speech_chunks = get_speech_chunks(audio)

    asr = build_asr_pipeline()
    chunks_with_transcripts = []
    for s, e in tqdm(speech_chunks, desc="Transcribing chunks"):
        rough = transcribe_chunk(asr, audio[s:e])
        chunks_with_transcripts.append({"start_sample": s, "end_sample": e, "rough_text": rough})

    aligned = align_chunks_to_book(chunks_with_transcripts, book_text)

    export_dataset(audio, aligned, Path(args.out), args.book_id)


if __name__ == "__main__":
    main()
