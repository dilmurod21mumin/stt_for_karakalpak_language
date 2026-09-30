"""
Download every video in a YouTube playlist as 16kHz mono WAV audio.

Usage:
    uv run python download_playlist.py \
        --url "https://www.youtube.com/playlist?list=XXXX" \
        --out /home/dilmurod/job/tts_project/data/raw_audio_books/book01
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_playlist(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # %(playlist_index)03d keeps files in playlist order, zero-padded,
    # so later scripts can sort them reliably and concatenate if needed.
    out_template = str(out_dir / "%(playlist_index)03d_%(title).60s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-x",                              # extract audio only
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",  # 16kHz mono
        "--yes-playlist",
        "--ignore-errors",                 # skip a broken video instead of aborting
        "--no-overwrites",
        "-o", out_template,
        url,
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("yt-dlp finished with a non-zero exit code. "
              "Some files may still have downloaded successfully; check the output folder.",
              file=sys.stderr)

    wavs = sorted(out_dir.glob("*.wav"))
    print(f"\nDownloaded {len(wavs)} WAV file(s) to {out_dir}")
    for w in wavs:
        print(" -", w.name)


def main():
    parser = argparse.ArgumentParser(description="Download a YouTube playlist as 16kHz mono WAV.")
    parser.add_argument("--url", required=True, help="YouTube playlist URL")
    parser.add_argument("--out", required=True, help="Output directory for WAV files")
    args = parser.parse_args()

    download_playlist(args.url, Path(args.out))


if __name__ == "__main__":
    main()
