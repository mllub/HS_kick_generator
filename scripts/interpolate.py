"""
Interpolate between two kicks in RAVE's latent space.

Encodes both kicks, linearly blends their latent representations at N steps,
and decodes each blend — great for creating hybrid or morphed kicks.

Usage:
    python scripts/interpolate.py \\
        --model outputs/kick_rave/kick_rave.ts \\
        --kick-a data/raw/kick1.wav \\
        --kick-b data/raw/kick2.wav \\
        --steps 8
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

try:
    import librosa
except ImportError:
    raise SystemExit("librosa required: pip install librosa")

TARGET_SR = 44100
TARGET_SAMPLES = int(TARGET_SR * 0.4)  # 17640


def load_model(model_path: str) -> torch.nn.Module:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def load_kick(path: str) -> torch.Tensor:
    """Load a kick, normalize, and return as [1, 1, T] tensor ready for RAVE encoding."""
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(audio) >= TARGET_SAMPLES:
        audio = audio[:TARGET_SAMPLES]
    else:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak
    return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]


def peak_normalize(audio: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return audio * (10 ** (target_dbfs / 20.0) / peak)


def main():
    parser = argparse.ArgumentParser(description="Latent-space interpolation between two kicks")
    parser.add_argument("--model", required=True, help="Exported .ts model path")
    parser.add_argument("--kick-a", required=True, help="First kick WAV")
    parser.add_argument("--kick-b", required=True, help="Second kick WAV")
    parser.add_argument("--steps", type=int, default=8,
                        help="Number of output files (includes both endpoints)")
    parser.add_argument("--output", default="outputs/interpolated", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    print(f"Encoding kicks...")
    kick_a = load_kick(args.kick_a)
    kick_b = load_kick(args.kick_b)

    with torch.no_grad():
        z_a = model.encode(kick_a)
        z_b = model.encode(kick_b)

    stem_a = Path(args.kick_a).stem
    stem_b = Path(args.kick_b).stem
    print(f"Generating {args.steps} interpolation steps ({stem_a} → {stem_b})...\n")

    alphas = np.linspace(0.0, 1.0, args.steps)
    for i, alpha in enumerate(alphas, 1):
        z = (1.0 - alpha) * z_a + alpha * z_b
        with torch.no_grad():
            audio = model.decode(z)
        audio = audio.squeeze().cpu().numpy()
        if len(audio) >= TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        else:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
        audio = peak_normalize(audio)

        fname = f"interp_{stem_a}_to_{stem_b}_{i:02d}of{args.steps}_a{alpha:.2f}.wav"
        sf.write(str(output_dir / fname), audio, TARGET_SR, subtype="PCM_24")
        print(f"  [{i:2d}/{args.steps}] α={alpha:.2f} → {fname}")

    print(f"\nDone. {args.steps} files in {output_dir}")


if __name__ == "__main__":
    main()
