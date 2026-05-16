import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

ModeType = Literal["ae", "vae_fixed", "vae"]


class KickVAE(nn.Module):
    """
    Convolutional VAE operating on 2D log-mel spectrograms of kick drums.

    mode:
        "ae"        — regular autoencoder; mu is used directly as latent code,
                      no KL term
        "vae_fixed" — VAE with fixed unit variance (logvar clamped to 0);
                      KL = 0.5 * sum(mu^2)
        "vae"       — full VAE; both mu and logvar are learned;
                      KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """

    def __init__(
        self,
        mode: ModeType = "vae",
        latent_dim: int = 32,
        n_mels: int = 128,
        n_frames: int = 69,
        enc_channels: list[int] = [32, 64, 128],
    ):
        super().__init__()
        assert mode in ("ae", "vae_fixed", "vae"), f"Unknown mode '{mode}'"
        self.mode = mode
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.n_frames = n_frames

        # --- Encoder conv blocks ---
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in enc_channels:
            enc_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.enc_conv = nn.Sequential(*enc_layers)

        # Compute flattened encoder output size via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, n_frames)
            enc_out = self.enc_conv(dummy)
            self._enc_shape = tuple(enc_out.shape[1:])   # (C, H, W)
            flat_size = enc_out.numel()

        self.fc_mu     = nn.Linear(flat_size, latent_dim)
        self.fc_logvar = nn.Linear(flat_size, latent_dim)

        # --- Decoder FC + conv blocks ---
        self.fc_dec = nn.Linear(latent_dim, flat_size)

        dec_in_channels = list(reversed(enc_channels))       # e.g. [128, 64, 32]
        dec_out_channels = dec_in_channels[1:] + [1]          # e.g. [64, 32, 1]

        dec_layers: list[nn.Module] = []
        for i, (in_c, out_c) in enumerate(zip(dec_in_channels, dec_out_channels)):
            is_last = i == len(dec_in_channels) - 1
            dec_layers.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2,
                                   padding=1, output_padding=1)
            )
            if not is_last:
                dec_layers += [nn.BatchNorm2d(out_c), nn.LeakyReLU(0.2, inplace=True)]
        self.dec_conv = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.mode == "ae" or not self.training:
            return mu
        if self.mode == "vae_fixed":
            return mu + torch.randn_like(mu)   # logvar ignored, unit gaussian
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), *self._enc_shape)
        out = self.dec_conv(h)
        # Resize to exact input shape — handles rounding differences from strided convs
        out = F.interpolate(out, size=(self.n_mels, self.n_frames),
                            mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.mode == "ae":
            return torch.zeros(1, device=mu.device).squeeze()
        if self.mode == "vae_fixed":
            # logvar exists but is ignored — KL against N(0,1) with unit variance
            return 0.5 * mu.pow(2).mean()
        # Full VAE: KL = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
