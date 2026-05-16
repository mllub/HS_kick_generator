"""
Detect and interactively review anomalous kick samples.

Scans a directory for noise-like or silent files using spectral flatness,
crest factor, and RMS energy. Then lets you audition and delete them.

Controls:
    SPACE  — play / replay sample
    D      — delete file, move to next
    K      — keep file, move to next

Usage:
    python scripts/review_samples.py
    python scripts/review_samples.py --input data/raw
    python scripts/review_samples.py --flatness 0.3 --crest 2.5 --silence -45
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import msvcrt
import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import librosa
except ImportError:
    print("Error: librosa not installed.")
    sys.exit(1)

TARGET_SR = 44100
EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}


def compute_scores(audio: np.ndarray) -> dict:
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    crest = peak / (rms + 1e-8)
    rms_db = 20 * np.log10(rms + 1e-8)
    return {"flatness": flatness, "crest": crest, "rms_db": rms_db}


def anomaly_reason(scores: dict, flatness_thresh: float, crest_thresh: float, silence_thresh: float) -> str:
    if scores["rms_db"] < silence_thresh:
        return f"silent (RMS {scores['rms_db']:.1f} dBFS)"
    if scores["flatness"] > flatness_thresh:
        return f"noise-like (flatness {scores['flatness']:.3f})"
    if scores["crest"] < crest_thresh:
        return f"low transient (crest {scores['crest']:.1f})"
    return ""


def play(audio: np.ndarray) -> None:
    sd.stop()
    sd.play(audio, samplerate=TARGET_SR)


def main():
    parser = argparse.ArgumentParser(description="Review anomalous kick samples")
    parser.add_argument("--input", default="data/raw", help="Directory to scan")
    parser.add_argument("--flatness", type=float, default=0.25,
                        help="Spectral flatness threshold — higher = more noise-like (default: 0.25)")
    parser.add_argument("--crest", type=float, default=3.0,
                        help="Minimum crest factor — lower = less transient (default: 3.0)")
    parser.add_argument("--silence", type=float, default=-40.0,
                        help="Silence threshold in dBFS (default: -40)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in EXTENSIONS)

    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    # --- Scan ---
    print(f"Scanning {len(files)} files for anomalies...")
    anomalies: list[tuple[Path, np.ndarray, str]] = []
    errors = 0
    for i, f in enumerate(files, 1):
        print(f"\r  [{i:4d}/{len(files)}] {f.name[:60]:<60}", end="", flush=True)
        try:
            audio, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
            scores = compute_scores(audio)
            reason = anomaly_reason(scores, args.flatness, args.crest, args.silence)
            if reason:
                anomalies.append((f, audio, reason))
        except Exception as e:
            anomalies.append((f, np.zeros(TARGET_SR, dtype=np.float32), f"load error: {e}"))
            errors += 1
    print()

    # --- List ---
    print(f"\nFound {len(anomalies)} anomalous sample(s):\n")
    for idx, (f, _, reason) in enumerate(anomalies, 1):
        print(f"  [{idx:3d}] {f.name:<50}  {reason}")

    if not anomalies:
        return

    print("\nControls: SPACE = play/replay   D = delete & next   K = keep & next\n")

    # --- Review loop ---
    deleted = 0
    kept = 0
    i = 0
    while i < len(anomalies):
        path, audio, reason = anomalies[i]
        exists = path.exists()
        status = "" if exists else "  [already deleted]"
        print(f"[{i + 1}/{len(anomalies)}] {path.name}  —  {reason}{status}")
        print("  > ", end="", flush=True)

        while True:
            key = msvcrt.getwch()
            if key == " ":
                if exists:
                    play(audio)
                    print("playing... ", end="", flush=True)
                else:
                    print("(file deleted) ", end="", flush=True)
            elif key.upper() == "D":
                sd.stop()
                if exists:
                    path.unlink()
                    deleted += 1
                    print(f"deleted.")
                else:
                    print(f"(already gone).")
                i += 1
                break
            elif key.upper() == "K":
                sd.stop()
                kept += 1
                print(f"kept.")
                i += 1
                break

    print(f"\nDone. {deleted} deleted, {kept} kept.")


if __name__ == "__main__":
    main()
