"""
Convert raw kick samples into a single kicks_prepared.npy file.

Does the slow part locally (librosa resample + onset detect + trim/pad +
normalize) so only one compact numpy array needs to be transferred.

On the remote, run:
    python scripts/preprocess.py --prepared data/prepared/kicks_prepared.npy

Usage:
    python scripts/prepare_for_ssh.py
    python scripts/prepare_for_ssh.py --input data/raw --output data/prepared
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    print("Error: librosa not installed. Run: pip install librosa")
    sys.exit(1)

TARGET_SR      = 44100
TARGET_SAMPLES = int(TARGET_SR * 0.45)


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


def _process_file(args: tuple[int, Path]) -> tuple[int, str, np.ndarray | None, str]:
    i, path = args
    try:
        audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
        onset = detect_onset_sample(audio, TARGET_SR)
        audio = audio[onset:]
        if len(audio) > TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        elif len(audio) < TARGET_SAMPLES:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)), mode="constant")
        audio = peak_normalize(audio).astype(np.float32)
        return i, path.name, audio, ""
    except Exception as exc:
        return i, path.name, None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Prepare raw kicks into a single .npy for SSH")
    parser.add_argument("--input",  default="data/raw",      help="Directory with raw audio files")
    parser.add_argument("--output", default="data/prepared", help="Output directory for kicks_prepared.npy")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in extensions)

    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    n_workers = os.cpu_count() or 1
    print(f"Found {len(files)} file(s) — processing with {n_workers} workers ...")
    print(f"  Target: {TARGET_SAMPLES} samples ({TARGET_SAMPLES / TARGET_SR:.2f} s) @ {TARGET_SR} Hz\n")

    worker_args = [(i, f) for i, f in enumerate(files, 1)]
    results: dict[int, np.ndarray | None] = {}
    errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_file, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            idx, fname, audio, err = future.result()
            if err:
                print(f"  [{idx:3d}/{len(files)}] ERROR {fname}: {err}")
                errors += 1
            else:
                print(f"  [{idx:3d}/{len(files)}] {fname}")
            results[idx] = audio

    # Reassemble in original file order
    all_audio = [results[i] for i in range(1, len(files) + 1) if results[i] is not None]

    if not all_audio:
        print("No samples processed successfully.")
        sys.exit(1)

    arr = np.stack(all_audio)   # (N, T) float32
    out_path = output_dir / "kicks_prepared.npy"
    np.save(str(out_path), arr)
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"\nSaved {len(all_audio)} samples ({errors} errors)")
    print(f"  Shape : {arr.shape}  dtype: {arr.dtype}")
    print(f"  File  : {out_path}  ({size_mb:.1f} MB)")
    print(f"\nNext steps:")
    print(f"  bash scripts/upload_prepared.sh")
    print(f"  # on remote:")
    print(f"  python scripts/preprocess.py --prepared data/prepared/kicks_prepared.npy")


if __name__ == "__main__":
    main()
