"""
Griffin-Lim vocoder: sample from a trained KickVAE and convert mel spectrograms to audio.

Pipeline:
    latent z  →  VAE decoder  →  log-mel (normalised)
    →  denormalise  →  exp  →  InverseMelScale  →  GriffinLim  →  WAV

Usage:
    # Generate 8 random samples from latest checkpoint
    python scripts/vocode.py

    # Generate from a specific checkpoint
    python scripts/vocode.py --checkpoint outputs/vae/kick_vae_ep0050.pt --n 8

    # Reconstruct training samples (encode then decode) instead of random sampling
    python scripts/vocode.py --reconstruct --n 8
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

import torch
import soundfile as sf
import torchaudio.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.kick_vae import KickVAE

TARGET_SR = 44100
N_MELS    = 128
N_FFT     = 1024
HOP       = 256
N_STFT    = N_FFT // 2 + 1
LOG_EPS   = 1e-9


def build_mel_transform(device: torch.device) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    ).to(device)


def compute_norm_stats(waveforms: torch.Tensor, mel_tf: T.MelSpectrogram, device: torch.device):
    """Compute the global s_min / s_max used during training normalisation."""
    waveforms = waveforms.to(device)
    with torch.no_grad():
        specs = mel_tf(waveforms.squeeze(1))
        log_specs = torch.log(specs + LOG_EPS).unsqueeze(1)
    return log_specs.min().item(), log_specs.max().item()


def denormalise(x: torch.Tensor, s_min: float, s_max: float) -> torch.Tensor:
    """Undo the [-1, 1] normalisation applied in train_vae.py → log-mel power."""
    return (x + 1) / 2 * (s_max - s_min + 1e-9) + s_min


def build_vocoder(mel_tf: T.MelSpectrogram, device: torch.device):
    fb      = mel_tf.mel_scale.fb                        # (n_stft, n_mels)
    fb_pinv = torch.linalg.pinv(fb.T).to(device)        # (n_stft, n_mels)

    griffin_lim = T.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP,
        power=2.0,
        n_iter=64,
    ).to(device)

    return fb_pinv, griffin_lim


def mel_to_audio(
    mel_norm: torch.Tensor,
    s_min: float,
    s_max: float,
    fb_pinv: torch.Tensor,
    griffin_lim: T.GriffinLim,
) -> torch.Tensor:
    """(1, 1, n_mels, n_frames) normalised log-mel  →  (T,) waveform."""
    log_mel   = denormalise(mel_norm, s_min, s_max)      # log amplitude
    mel_power = torch.exp(log_mel).squeeze(0)            # (1, n_mels, n_frames) power
    linear    = torch.relu(fb_pinv @ mel_power)          # (1, n_stft, n_frames)
    wav       = griffin_lim(linear)                      # (1, T)
    return wav.squeeze(0)                                # (T,)


def main():
    parser = argparse.ArgumentParser(description="Griffin-Lim vocoder for KickVAE")
    parser.add_argument("--checkpoint",  default=None,              help="Path to checkpoint (.pt). Defaults to latest in --ckpt-dir.")
    parser.add_argument("--ckpt-dir",    default="outputs/vae",     help="Directory to search for latest checkpoint")
    parser.add_argument("--data",        default="data/processed/kicks.pt", help="kicks.pt — used for norm stats and reconstruction mode")
    parser.add_argument("--out",         default="outputs/audio",   help="Output directory for WAV files")
    parser.add_argument("--n",           type=int, default=8,       help="Number of samples to generate")
    parser.add_argument("--latent-dim",  type=int, default=128)
    parser.add_argument("--reconstruct", action="store_true",       help="Encode then decode training samples instead of random sampling")
    parser.add_argument("--mode",        default=None,              choices=["ae", "vae_fixed", "vae"],
                        help="Override mode (default: read from checkpoint)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Resolve checkpoint ─────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_dir   = Path(args.ckpt_dir)
        candidates = sorted(ckpt_dir.glob("kick_vae_ep*.pt"))
        final      = ckpt_dir / "kick_vae_final.pt"
        if final.exists():
            ckpt_path = final
        elif candidates:
            ckpt_path = candidates[-1]
        else:
            print(f"No checkpoints found in {args.ckpt_dir}")
            sys.exit(1)
    print(f"Checkpoint : {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        saved_args = ckpt.get("args", {})
        mode       = args.mode or saved_args.get("mode", "vae")
        latent_dim = saved_args.get("latent_dim", args.latent_dim)
        state_dict = ckpt["model_state"]
    else:
        mode       = args.mode or "vae"
        latent_dim = args.latent_dim
        state_dict = ckpt
    print(f"Mode       : {mode}  |  Latent dim : {latent_dim}")

    # ── Load data for normalisation stats ─────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}  (needed for normalisation stats)")
        sys.exit(1)
    waveforms = torch.load(data_path, weights_only=True)   # (N, 1, T)
    n_frames  = None

    mel_tf = build_mel_transform(device)
    print("Computing normalisation stats from dataset ...")
    s_min, s_max = compute_norm_stats(waveforms, mel_tf, device)
    print(f"  s_min={s_min:.4f}  s_max={s_max:.4f}")

    # Determine n_frames from a sample spectrogram
    with torch.no_grad():
        sample_spec = mel_tf(waveforms[:1].to(device).squeeze(1))
    n_frames = sample_spec.shape[-1]
    print(f"  n_frames : {n_frames}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = KickVAE(mode=mode, latent_dim=latent_dim, n_mels=N_MELS, n_frames=n_frames).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Vocoder ───────────────────────────────────────────────────────────
    fb_pinv, griffin_lim = build_vocoder(mel_tf, device)

    # ── Generate ──────────────────────────────────────────────────────────
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def vocode_batch(mel_batch: torch.Tensor, tag: str) -> None:
        print(f"\nVocoding {len(mel_batch)} {tag} samples ...")
        for i in range(len(mel_batch)):
            wav  = mel_to_audio(mel_batch[i].unsqueeze(0), s_min, s_max, fb_pinv, griffin_lim)
            peak = wav.abs().max().clamp(min=1e-9)
            wav  = wav / peak * 0.9
            out_path = out_dir / f"kick_{tag}_{i:03d}.wav"
            sf.write(str(out_path), wav.cpu().numpy(), TARGET_SR)
            print(f"  Saved {out_path}")

    with torch.no_grad():
        # Pick 8 random dataset samples — used for both orig and recon
        indices = torch.randperm(len(waveforms))[: args.n]
        batch   = waveforms[indices].to(device)

        # Save originals
        print(f"\nSaving {args.n} original samples ...")
        for i, wav in enumerate(batch):
            audio = wav.squeeze(0).cpu().numpy()
            peak  = abs(audio).max().clip(1e-9)
            sf.write(str(out_dir / f"kick_orig_{i:03d}.wav"), audio / peak * 0.9, TARGET_SR)
            print(f"  Saved kick_orig_{i:03d}.wav")

        # Reconstruction: encode → decode the same samples
        log_mel = torch.log(mel_tf(batch.squeeze(1)) + LOG_EPS).unsqueeze(1)
        log_mel = 2 * (log_mel - s_min) / (s_max - s_min + 1e-9) - 1
        mel_recon, _, _ = model(log_mel)
        vocode_batch(mel_recon, "recon")

        # Random generation: z ~ N(0, I)
        z       = torch.randn(args.n, latent_dim, device=device)
        mel_gen = model.decode(z)
        vocode_batch(mel_gen, "gen")

    print(f"\nDone. WAV files written to {out_dir}/")


if __name__ == "__main__":
    main()
