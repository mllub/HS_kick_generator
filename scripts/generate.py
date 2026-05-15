"""
Generate hardstyle kicks from a trained and exported RAVE model.

Prerequisites: export the model first with:
    python scripts/export.py --run outputs/kick_rave

Usage:
    python scripts/generate.py --model outputs/kick_rave/kick_rave.ts
    python scripts/generate.py --model outputs/kick_rave/kick_rave.ts --num 20 --temperature 0.9
    python scripts/generate.py --model outputs/kick_rave/kick_rave.ts --temperature 1.3 --seed 42
"""
import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

TARGET_SR = 44100
TARGET_DURATION = 0.4
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 17640

# These must match kick_rave.gin:
#   N_BAND=16, RATIOS=[4,4,2] → total hop = 16 × 4 × 4 × 2 = 512
DEFAULT_HOP = 512
DEFAULT_LATENT_SIZE = 8


def load_model(model_path: str) -> torch.nn.Module:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def get_model_params(model) -> tuple:
    """Read sr and hop length from exported model attributes where available."""
    try:
        sr = int(model.sr)
    except Exception:
        sr = TARGET_SR
        print(f"  (sr not found on model, defaulting to {TARGET_SR})")

    try:
        # RAVE exported models expose encode_params = [latent_size, hop_length, ...]
        encode_params = model.encode_params
        hop = int(encode_params[1])
    except Exception:
        hop = DEFAULT_HOP
        print(f"  (hop not found on model, defaulting to {DEFAULT_HOP} — matches kick_rave.gin)")

    return sr, hop


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio * (10 ** (target_dbfs / 20.0) / peak)


def generate_kick(
    model,
    latent_size: int,
    n_frames: int,
    temperature: float,
    target_samples: int,
) -> np.ndarray:
    with torch.no_grad():
        z = torch.randn(1, latent_size, n_frames) * temperature
        audio = model.decode(z)
    audio = audio.squeeze().cpu().numpy()
    if len(audio) >= target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - len(audio)))


def main():
    parser = argparse.ArgumentParser(description="Generate kicks from a trained RAVE model")
    parser.add_argument("--model", required=True, help="Path to exported .ts model file")
    parser.add_argument("--num", type=int, default=10, help="Number of kicks to generate")
    parser.add_argument("--output", default="outputs/generated", help="Output directory")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature: 0.5=conservative, 1.0=normal, 1.3+=experimental",
    )
    parser.add_argument(
        "--latent-size", type=int, default=DEFAULT_LATENT_SIZE,
        help=f"Latent size (default: {DEFAULT_LATENT_SIZE}, matches kick_rave.gin LATENT_SIZE)",
    )
    parser.add_argument(
        "--hop", type=int, default=DEFAULT_HOP,
        help=f"Model hop length in samples (default: {DEFAULT_HOP}, matches kick_rave.gin N_BAND × RATIOS)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Export first: python scripts/export.py --run outputs/kick_rave")
        raise SystemExit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = load_model(str(model_path))
    sr, hop = get_model_params(model)

    # +1 frame to ensure we always have enough samples after decoding
    n_frames = math.ceil(TARGET_SAMPLES / hop) + 1

    print(f"  SR={sr} Hz  hop={hop}  frames/kick={n_frames}  latent={args.latent_size}")
    print(f"  Temperature={args.temperature}  Generating {args.num} kicks → {output_dir}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(1, args.num + 1):
        audio = generate_kick(model, args.latent_size, n_frames, args.temperature, TARGET_SAMPLES)
        audio = peak_normalize(audio)
        out_path = output_dir / f"kick_{timestamp}_{i:03d}.wav"
        sf.write(str(out_path), audio, sr, subtype="PCM_24")
        print(f"  [{i:3d}/{args.num}] {out_path.name}")

    print(f"\nDone. {args.num} kicks saved to {output_dir}")


if __name__ == "__main__":
    main()
