"""
Audition all samples in kicks.pt: view mel spectrogram and compare original
audio against Griffin-Lim reconstruction.

Controls:
    SPACE       — play Griffin-Lim reconstruction
    O           — play original waveform
    LEFT / RIGHT — previous / next sample
    Q           — quit

Usage:
    python scripts/mel_audition.py
    python scripts/mel_audition.py --data data/processed/kicks.pt
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio.transforms as T

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice not installed.  pip install sounddevice")
    sys.exit(1)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

TARGET_SR = 44100
N_MELS    = 128
N_FFT     = 1024
HOP       = 256
N_STFT    = N_FFT // 2 + 1
LOG_EPS   = 1e-9


# ── Transforms ────────────────────────────────────────────────────────────────

def build_transforms(device: torch.device):
    mel_tf = T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    ).to(device)

    # Pseudo-inverse of the mel filterbank: more robust than InverseMelScale
    # fb shape: (n_stft, n_mels); pinv(fb.T) maps mel → linear
    fb      = mel_tf.mel_scale.fb                        # (n_stft, n_mels)
    fb_pinv = torch.linalg.pinv(fb.T).to(device)        # (n_stft, n_mels)

    griffin_lim = T.GriffinLim(
        n_fft=N_FFT,
        hop_length=HOP,
        power=2.0,
        n_iter=64,
    ).to(device)

    return mel_tf, fb_pinv, griffin_lim


def to_log_mel_norm(waveform: torch.Tensor, mel_tf, s_min: float, s_max: float) -> torch.Tensor:
    """(1, T) → (1, n_mels, n_frames) normalised log-mel, matching train_vae.py."""
    spec = mel_tf(waveform)                              # (1, n_mels, n_frames)
    log  = torch.log(spec + LOG_EPS)
    norm = 2 * (log - s_min) / (s_max - s_min + 1e-9) - 1
    return norm


def reconstruct(mel_norm: torch.Tensor, s_min: float, s_max: float,
                fb_pinv: torch.Tensor, griffin_lim) -> np.ndarray:
    """(1, n_mels, n_frames) normalised log-mel → numpy waveform."""
    log_mel   = (mel_norm + 1) / 2 * (s_max - s_min + 1e-9) + s_min
    mel_power = torch.exp(log_mel)                       # (1, n_mels, n_frames)
    # fb_pinv: (n_stft, n_mels);  mel_power: (1, n_mels, n_frames)
    linear    = torch.relu(fb_pinv @ mel_power)          # (1, n_stft, n_frames)
    wav       = griffin_lim(linear).squeeze(0)           # (T,)
    peak      = wav.abs().max().clamp(min=1e-9)
    return (wav / peak * 0.9).cpu().numpy()


def play(audio: np.ndarray) -> None:
    sd.stop()
    sd.play(audio.astype(np.float32), samplerate=TARGET_SR)


# ── Viewer ────────────────────────────────────────────────────────────────────

class MelAuditioner:
    def __init__(self, waveforms: torch.Tensor, s_min: float, s_max: float,
                 mel_tf, fb_pinv, griffin_lim, device: torch.device):
        self.waveforms   = waveforms          # (N, 1, T)
        self.s_min       = s_min
        self.s_max       = s_max
        self.mel_tf      = mel_tf
        self.fb_pinv     = fb_pinv
        self.griffin_lim = griffin_lim
        self.device      = device
        self.idx         = 0
        self.n           = len(waveforms)

        # Cache: index → (mel_norm cpu tensor, recon numpy)
        self._mel_cache:   dict[int, torch.Tensor] = {}
        self._recon_cache: dict[int, np.ndarray]   = {}

        self.fig, (self.ax_orig, self.ax_mel, self.ax_recon) = plt.subplots(
            1, 3, figsize=(16, 4)
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        plt.tight_layout()
        plt.show()

    # ── cache helpers ──────────────────────────────────────────────────────

    def _get_mel(self, i: int) -> torch.Tensor:
        if i not in self._mel_cache:
            wav = self.waveforms[i].to(self.device)      # (1, T)
            with torch.no_grad():
                self._mel_cache[i] = to_log_mel_norm(wav, self.mel_tf,
                                                     self.s_min, self.s_max).cpu()
        return self._mel_cache[i]

    def _get_recon(self, i: int) -> np.ndarray:
        if i not in self._recon_cache:
            mel = self._get_mel(i).to(self.device)
            with torch.no_grad():
                self._recon_cache[i] = reconstruct(mel, self.s_min, self.s_max,
                                                   self.fb_pinv, self.griffin_lim)
        return self._recon_cache[i]

    # ── drawing ───────────────────────────────────────────────────────────

    def _draw(self) -> None:
        i   = self.idx
        wav = self.waveforms[i, 0].numpy()               # (T,)
        mel = self._get_mel(i).squeeze().numpy()         # (n_mels, n_frames)
        t   = np.arange(len(wav)) / TARGET_SR

        self.ax_orig.cla()
        self.ax_orig.plot(t, wav, linewidth=0.4, color="steelblue")
        self.ax_orig.set_title(f"Original waveform  [{i+1}/{self.n}]")
        self.ax_orig.set_xlabel("Time (s)")
        self.ax_orig.set_ylim(-1.1, 1.1)

        self.ax_mel.cla()
        self.ax_mel.imshow(mel, origin="lower", aspect="auto", cmap="magma")
        self.ax_mel.set_title("Log-mel (network input)")
        self.ax_mel.set_xlabel("Frame")
        self.ax_mel.set_ylabel("Mel bin")

        recon = self._get_recon(i)
        t_r   = np.arange(len(recon)) / TARGET_SR
        self.ax_recon.cla()
        self.ax_recon.plot(t_r, recon, linewidth=0.4, color="tomato")
        self.ax_recon.set_title("Griffin-Lim reconstruction")
        self.ax_recon.set_xlabel("Time (s)")
        self.ax_recon.set_ylim(-1.1, 1.1)

        self.fig.suptitle(
            f"Sample {i+1} / {self.n}   |   "
            "SPACE = play recon   O = play original   ← → = navigate   Q = quit",
            fontsize=9,
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    # ── key handler ───────────────────────────────────────────────────────

    def _on_key(self, event) -> None:
        key = event.key

        if key == " ":
            play(self._get_recon(self.idx))

        elif key.lower() == "o":
            play(self.waveforms[self.idx, 0].numpy())

        elif key == "right":
            self.idx = (self.idx + 1) % self.n
            self._draw()

        elif key == "left":
            self.idx = (self.idx - 1) % self.n
            self._draw()

        elif key.lower() == "q":
            sd.stop()
            plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audition mel reconstructions")
    parser.add_argument("--data", default="data/processed/kicks.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    print(f"Loading {data_path} ...")
    waveforms = torch.load(data_path, weights_only=True)   # (N, 1, T)
    print(f"  {len(waveforms)} samples")

    mel_tf, fb_pinv, griffin_lim = build_transforms(device)

    # Compute global normalisation stats (same as train_vae.py)
    print("Computing normalisation stats ...")
    with torch.no_grad():
        specs    = mel_tf(waveforms.to(device).squeeze(1))   # (N, n_mels, n_frames)
        log_specs = torch.log(specs + LOG_EPS)
        s_min    = log_specs.min().item()
        s_max    = log_specs.max().item()
    waveforms = waveforms.cpu()
    print(f"  s_min={s_min:.4f}  s_max={s_max:.4f}")
    print("\nControls: SPACE=play recon  O=play original  ←/→=navigate  Q=quit\n")

    MelAuditioner(waveforms, s_min, s_max, mel_tf, fb_pinv, griffin_lim, device)


if __name__ == "__main__":
    main()
