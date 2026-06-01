import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

ModeType = Literal["ae", "vae_fixed", "vae"]


# ── Building blocks ───────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Pre-activation residual block: (Norm→SiLU→Conv)×2 + skip connection.

    When in_ch != out_ch a 1×1 conv aligns the shortcut; otherwise it is a
    plain identity wire with no extra parameters.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


# ── KickVAE ───────────────────────────────────────────────────────────────────

class KickVAE(nn.Module):
    """
    Fully-convolutional VAE on 2D log-mel spectrograms, with residual blocks.

    Each encoder stage: stride-2 Conv (downsample) → ResBlock.
    Each decoder stage: stride-2 ConvTranspose (upsample) → ResBlock.
    Bottleneck: 1×1 conv_mu / conv_logvar → latent z (spatial tensor) → conv_dec.

    Using kernel=4, stride=2, padding=1 gives exact ×2 spatial scaling with no
    rounding artefacts, removing the need for the bilinear resize at the end.

    mode:
        "ae"        — AE; mu used directly as z, no KL term.
        "vae_fixed" — VAE with unit variance; KL = 0.5 * mean(mu²).
        "vae"       — full VAE; both mu and logvar learned.
    """

    def __init__(
        self,
        mode: ModeType = "vae",
        latent_dim: int = 64,
        n_mels: int = 128,
        n_frames: int = 69,
        enc_channels: list[int] = [64, 128, 256, 512],
        dropout: float = 0.05,
    ):
        super().__init__()
        assert mode in ("ae", "vae_fixed", "vae"), f"Unknown mode '{mode}'"
        self.mode = mode
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.n_frames = n_frames

        # ── Encoder ───────────────────────────────────────────────────────
        # Each stage: stride-2 conv (channel change + downsample) + ResBlock
        enc_stages: list[nn.Module] = []
        in_ch = 1
        for out_ch in enc_channels:
            enc_stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                _ResBlock(out_ch, out_ch, dropout),
            ))
            in_ch = out_ch
        self.enc_stages = nn.ModuleList(enc_stages)

        # ── 1×1 conv bottleneck ───────────────────────────────────────────
        self.conv_mu     = nn.Conv2d(enc_channels[-1], latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(enc_channels[-1], latent_dim, kernel_size=1)
        self.conv_dec    = nn.Conv2d(latent_dim, enc_channels[-1], kernel_size=1)

        # ── Decoder ───────────────────────────────────────────────────────
        # Each stage: stride-2 ConvTranspose (upsample) + ResBlock
        # Final stage goes to 1 channel (output) — no ResBlock on that layer.
        dec_in_ch  = list(reversed(enc_channels))        # [512, 256, 128, 64]
        dec_out_ch = dec_in_ch[1:] + [enc_channels[0]]   # [256, 128,  64, 64]

        dec_stages: list[nn.Module] = []
        for in_c, out_c in zip(dec_in_ch, dec_out_ch):
            dec_stages.append(nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                _ResBlock(out_c, out_c, dropout),
            ))
        self.dec_stages = nn.ModuleList(dec_stages)

        # Final projection to 1 output channel
        self.dec_out = nn.Conv2d(enc_channels[0], 1, kernel_size=3, padding=1)

    # ── Forward passes ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for stage in self.enc_stages:
            h = stage(h)
        return self.conv_mu(h), self.conv_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.mode == "ae" or not self.training:
            return mu
        if self.mode == "vae_fixed":
            return mu + torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_dec(z)
        for stage in self.dec_stages:
            h = stage(h)
        out = self.dec_out(h)
        # Bilinear resize handles any residual rounding from the conv chain
        return F.interpolate(out, size=(self.n_mels, self.n_frames),
                             mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.mode == "ae":
            return torch.zeros(1, device=mu.device).squeeze()
        if self.mode == "vae_fixed":
            return 0.5 * mu.pow(2).mean()
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
