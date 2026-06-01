import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

ModeType = Literal["ae", "vae_fixed", "vae"]


class KickVAE(nn.Module):
    """
    Fully-convolutional VAE on 2D log-mel spectrograms.

    The bottleneck is a pair of 1×1 convolutions (conv_mu / conv_logvar) that
    map the deepest encoder feature map to *latent_dim* channels — no flatten,
    no linear layer.  The latent code z therefore retains spatial structure:
        shape  (B, latent_dim, H_bot, W_bot)
    where H_bot × W_bot is the feature-map size after all stride-2 encoder
    convolutions.  conv_dec maps z back to enc_channels[-1] channels before
    the transposed-conv decoder stack.

    mode:
        "ae"        — AE; mu used directly as z, no KL term.
        "vae_fixed" — VAE with unit variance; KL = 0.5 * mean(mu²).
        "vae"       — full VAE; both mu and logvar learned.
    """

    def __init__(
        self,
        mode: ModeType = "vae",
        latent_dim: int = 64,           # latent channels at bottleneck
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
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in enc_channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=10, stride=2, padding=4),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
                nn.Dropout2d(dropout),
            ]
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc_layers)

        # ── 1×1 conv bottleneck (fully convolutional, no FC) ──────────────
        self.conv_mu     = nn.Conv2d(enc_channels[-1], latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(enc_channels[-1], latent_dim, kernel_size=1)
        self.conv_dec    = nn.Conv2d(latent_dim, enc_channels[-1], kernel_size=1)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_in_ch  = list(reversed(enc_channels))   # [512, 256, 128, 64]
        dec_out_ch = dec_in_ch[1:] + [1]            # [256, 128, 64,   1]

        dec_layers: list[nn.Module] = []
        for i, (in_c, out_c) in enumerate(zip(dec_in_ch, dec_out_ch)):
            is_last = i == len(dec_in_ch) - 1
            dec_layers.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=10, stride=2,
                                   padding=4, output_padding=1)
            )
            if not is_last:
                dec_layers += [
                    nn.BatchNorm2d(out_c),
                    nn.SiLU(),
                    nn.Dropout2d(dropout),
                ]
        self.dec_conv = nn.Sequential(*dec_layers)

    # ── Forward passes ────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_conv(x)                    # (B, C_enc, H_bot, W_bot)
        return self.conv_mu(h), self.conv_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.mode == "ae" or not self.training:
            return mu
        if self.mode == "vae_fixed":
            return mu + torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_dec(z)                    # (B, C_enc, H_bot, W_bot)
        out = self.dec_conv(h)
        # Bilinear resize handles any rounding from strided convolutions
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
