"""
Preprocess raw kick samples for training.

Steps per file:
  1. Resample to 44100 Hz, convert to mono
  2. Detect the first onset, align audio to it (2 ms pre-roll to preserve attack)
  3. Trim or zero-pad to exactly 0.4 s (17640 samples)
  4. Peak-normalize to -1 dBFS
  5. Save 24-bit WAV to data/processed/
  6. Generate augmented variants (polarity flip + bell-filtered copies)

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --input my_kicks/ --processed data/processed/
    python scripts/preprocess.py --no-augment
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import sosfilt

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


def bell_filter(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Peaking (bell) EQ biquad — Audio EQ Cookbook."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, audio)


_N_EQ_VARIANTS = 5
_N_BELL_BANDS = 4       # bells per EQ realization
_FREQ_MIN = 20.0        # Hz — log-sampled so all octaves get equal coverage
_FREQ_MAX = 20000.0
_GAIN_RANGE = 7.0       # ±dB
_Q_MIN, _Q_MAX = 0.5, 3.0
_CLIP_THRESHOLD = 0.8   # fraction of peak before normalization


def _random_eq(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Apply _N_BELL_BANDS random peaking filters log-uniformly across 20–20k Hz."""
    out = audio.copy()
    freqs = np.exp(rng.uniform(np.log(_FREQ_MIN), np.log(_FREQ_MAX), size=_N_BELL_BANDS))
    for freq in freqs:
        gain = rng.uniform(-_GAIN_RANGE, _GAIN_RANGE)
        q = rng.uniform(_Q_MIN, _Q_MAX)
        out = bell_filter(out, sr, float(freq), float(gain), float(q))
    return peak_normalize(out)


def _clip_distort(audio: np.ndarray) -> np.ndarray:
    """Hard clip at 80% of the signal's peak, then re-normalize."""
    threshold = _CLIP_THRESHOLD * np.max(np.abs(audio))
    return peak_normalize(np.clip(audio, -threshold, threshold))


def augment(audio: np.ndarray, sr: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """Return (suffix, audio) pairs for each augmented variant.

    Per input file:
      - 5 EQ realizations of the original + clipped version of each
      - 1 polarity flip
      - 5 EQ realizations of the polarity flip + clipped version of each
    Total: 21 augmented files per source sample.
    """
    variants: list[tuple[str, np.ndarray]] = []

    polarity = -audio

    for base, prefix in [(audio, "_aug_eq"), (polarity, "_aug_pol_eq")]:
        if prefix == "_aug_pol_eq":
            variants.append(("_aug_pol", polarity))
        for i in range(_N_EQ_VARIANTS):
            eq_audio = _random_eq(base, sr, rng)
            variants.append((f"{prefix}{i + 1}", eq_audio))
            variants.append((f"{prefix}{i + 1}_clip", _clip_distort(eq_audio)))

    return variants


def process_audio(input_path: Path, target_samples: int) -> np.ndarray:
    audio, _ = librosa.load(str(input_path), sr=TARGET_SR, mono=True)
    onset = detect_onset_sample(audio, TARGET_SR)
    audio = audio[onset:]
    if len(audio) >= target_samples:
        audio = audio[:target_samples]
    else:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode="constant")
    return peak_normalize(audio)


def main():
    parser = argparse.ArgumentParser(description="Preprocess kick samples for training")
    parser.add_argument("--input", default="data/raw", help="Directory with raw kick samples")
    parser.add_argument("--processed", default="data/processed", help="Output dir for processed WAVs")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip data augmentation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for augmentation")
    args = parser.parse_args()

    input_dir = Path(args.input)
    processed_dir = Path(args.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in extensions)

    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s) in {input_dir}")

    while True:
        raw = input("Sample duration in seconds (e.g. 0.4): ").strip()
        try:
            duration = float(raw)
            if duration <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Please enter a positive number.")
    target_samples = int(TARGET_SR * duration)

    if not args.no_augment:
        answer = input("Run augmentation? (21 variants per file: EQ, clipping, polarity) [y/n]: ").strip().lower()
        if answer != "y":
            args.no_augment = True

    save_wavs = input("Save processed WAV files to disk? [y/n]: ").strip().lower() == "y"

    aug_label = "disabled" if args.no_augment else "21 variants per file (5 EQ + 5 EQ+clip + polarity + 5 pol EQ + 5 pol EQ+clip)"
    print(f"Processing {len(files)} files → {target_samples} samples ({duration} s) @ {TARGET_SR} Hz  (augmentation: {aug_label})")
    errors = 0
    all_audio = []
    for i, f in enumerate(files, 1):
        try:
            audio = process_audio(f, target_samples)
            print(f"  [{i:3d}/{len(files)}] {f.name}")
            all_audio.append(audio)

            if save_wavs:
                sf.write(str(processed_dir / f"{f.stem}.wav"), audio, TARGET_SR, subtype="PCM_24")

            if not args.no_augment:
                file_rng = np.random.default_rng(args.seed + i)
                for suffix, aug_audio in augment(audio, TARGET_SR, file_rng):
                    all_audio.append(aug_audio)
                    if save_wavs:
                        sf.write(str(processed_dir / f"{f.stem}{suffix}.wav"), aug_audio, TARGET_SR, subtype="PCM_24")
        except Exception as e:
            print(f"  [{i:3d}/{len(files)}] ERROR {f.name}: {e}")
            errors += 1

    processed = len(files) - errors
    aug_count = 0 if args.no_augment else processed * 21
    print(f"\nDone. {processed} processed, {errors} errors (+{aug_count} augmented) — {len(all_audio)} total")

    if not all_audio:
        return

    tensor = torch.tensor(np.stack(all_audio), dtype=torch.float32).unsqueeze(1)  # (N, 1, T)
    out_path = processed_dir / "kicks.pt"
    torch.save(tensor, out_path)
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"\nTensor saved:")
    print(f"  Shape : {tuple(tensor.shape)}  ({tensor.shape[0]} samples × {tensor.shape[1]} frames)")
    print(f"  Size  : {size_mb:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
