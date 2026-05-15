"""
Preprocess raw kick samples for RAVE training.

Steps per file:
  1. Resample to 44100 Hz, convert to mono
  2. Detect the first onset, align audio to it (2 ms pre-roll to preserve attack)
  3. Trim or zero-pad to exactly 0.4 s (17640 samples)
  4. Peak-normalize to -1 dBFS
  5. Save 24-bit WAV to data/processed/

Then calls `rave preprocess` to build the LMDB training database.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input my_kicks/ --db data/mykicks.mdb
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: pip install librosa")
    sys.exit(1)

TARGET_SR = 44100
TARGET_SAMPLES = int(TARGET_SR * 0.4)  # 17640 samples


def detect_onset_sample(audio: np.ndarray, sr: int, pre_roll_ms: float = 2.0) -> int:
    onset_frames = librosa.onset.onset_detect(
        y=audio, sr=sr, units="samples", hop_length=64
    )
    if len(onset_frames) == 0:
        return 0
    pre_roll = int(sr * pre_roll_ms / 1000)
    return max(0, int(onset_frames[0]) - pre_roll)


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio * (10 ** (target_dbfs / 20.0) / peak)


def process_file(input_path: Path, output_path: Path) -> None:
    audio, _ = librosa.load(str(input_path), sr=TARGET_SR, mono=True)

    onset = detect_onset_sample(audio, TARGET_SR)
    audio = audio[onset:]

    if len(audio) >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")

    audio = peak_normalize(audio)
    sf.write(str(output_path), audio, TARGET_SR, subtype="PCM_24")


def main():
    parser = argparse.ArgumentParser(description="Preprocess kick samples for RAVE training")
    parser.add_argument("--input", default="data/raw", help="Directory with raw kick samples")
    parser.add_argument("--processed", default="data/processed", help="Output dir for processed WAVs")
    parser.add_argument("--db", default="data/kicks.mdb", help="LMDB path for RAVE training database")
    parser.add_argument("--skip-rave-preprocess", action="store_true",
                        help="Only normalize/pad; skip the rave preprocess step")
    args = parser.parse_args()

    input_dir = Path(args.input)
    processed_dir = Path(args.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in extensions)

    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(files)} files → {TARGET_SAMPLES} samples @ {TARGET_SR} Hz")
    errors = 0
    for i, f in enumerate(files, 1):
        out = processed_dir / f"{f.stem}.wav"
        try:
            process_file(f, out)
            print(f"  [{i:3d}/{len(files)}] {f.name}")
        except Exception as e:
            print(f"  [{i:3d}/{len(files)}] ERROR {f.name}: {e}")
            errors += 1

    print(f"\nDone. {len(files) - errors}/{len(files)} OK → {processed_dir}")

    if args.skip_rave_preprocess:
        return

    print(f"\nRunning rave preprocess → {args.db}")
    cmd = [
        "rave", "preprocess",
        "--input_path", str(processed_dir),
        "--output_path", args.db,
        "--sampling_rate", str(TARGET_SR),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"RAVE preprocessing complete → {args.db}")
    except FileNotFoundError:
        print("Warning: 'rave' CLI not found in PATH. Run manually:")
        print(f"  rave preprocess --input_path {processed_dir} --output_path {args.db} --sampling_rate {TARGET_SR}")
    except subprocess.CalledProcessError as e:
        print(f"rave preprocess failed with exit code {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
