"""Pipeline: multi-channel mixer with master bus processing."""

from __future__ import annotations

import math

import numpy as np

from .blocks import Limiter
from .channel import Channel


class Pipeline:
    """Multi-channel DSP pipeline that renders a stereo mix.

    Each :class:`~kickgen.channel.Channel` is rendered to mono, panned to
    stereo using equal-power panning, and summed.  A master :class:`~kickgen.blocks.Limiter`
    protects the output.

    Parameters
    ----------
    channels:
        Ordered list of ``(name, channel)`` pairs.
    master_gain_db:
        Master output gain in dB, applied before the limiter.
    use_limiter:
        Whether to apply the master limiter.
    """

    def __init__(
        self,
        channels: list[tuple[str, Channel]],
        master_gain_db: float = 0.0,
        use_limiter: bool = True,
    ) -> None:
        self.channels: list[tuple[str, Channel]] = list(channels)
        self.master_gain_db = float(master_gain_db)
        self.use_limiter = bool(use_limiter)
        self.master_limiter = Limiter()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, length_seconds: float, sr: int = 44100) -> np.ndarray:
        """Render the full pipeline to a stereo float32 array.

        Parameters
        ----------
        length_seconds:
            Duration of the rendered output in seconds.
        sr:
            Sample rate in Hz (default 44100).

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, 2)``, dtype float32, stereo audio.
        """
        n_samples = int(length_seconds * sr)
        stereo = np.zeros((n_samples, 2), dtype=np.float32)

        for _name, channel in self.channels:
            mono = channel.process(sr, n_samples)  # (n_samples,)

            # Equal-power pan: pan in [-1, 1]
            pan_norm = (channel.pan + 1.0) / 2.0  # map to [0, 1]
            left_gain = math.cos(pan_norm * math.pi / 2.0)
            right_gain = math.sin(pan_norm * math.pi / 2.0)

            stereo[:, 0] += mono * left_gain
            stereo[:, 1] += mono * right_gain

        # Master gain
        master_linear = 10.0 ** (self.master_gain_db / 20.0)
        stereo = stereo * master_linear

        # Master limiter applied per channel
        if self.use_limiter:
            stereo[:, 0] = self.master_limiter.process(stereo[:, 0], sr)
            stereo[:, 1] = self.master_limiter.process(stereo[:, 1], sr)

        return stereo.astype(np.float32)

    # ------------------------------------------------------------------
    # Parameter interface
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, float]:
        """Return flat param dict with keys like ``"channel_name.block_name.param_name"``."""
        params: dict[str, float] = {
            "master_gain_db": self.master_gain_db,
        }
        for name, channel in self.channels:
            for k, v in channel.get_params().items():
                params[f"{name}.{k}"] = v
        for k, v in self.master_limiter.get_params().items():
            params[f"master_limiter.{k}"] = v
        return params

    def set_params(self, **kwargs: float) -> None:
        """Set parameters by flat key ``"channel_name.block_name.param_name"``."""
        for key, value in kwargs.items():
            if key == "master_gain_db":
                self.master_gain_db = float(value)
                continue

            # Split on first '.' only
            dot = key.find(".")
            if dot < 0:
                continue
            prefix = key[:dot]
            rest = key[dot + 1:]

            if prefix == "master_limiter":
                self.master_limiter.set_params(**{rest: value})
            else:
                for name, channel in self.channels:
                    if name == prefix:
                        channel.set_params(**{rest: value})
                        break

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for all parameters."""
        bounds: dict[str, tuple[float, float]] = {
            "master_gain_db": (-40.0, 12.0),
        }
        for name, channel in self.channels:
            for k, v in channel.param_bounds().items():
                bounds[f"{name}.{k}"] = v
        for k, v in self.master_limiter.param_bounds().items():
            bounds[f"master_limiter.{k}"] = v
        return bounds

    def __repr__(self) -> str:
        ch_names = [n for n, _ in self.channels]
        return (
            f"Pipeline(channels={ch_names}, master_gain_db={self.master_gain_db}, "
            f"use_limiter={self.use_limiter})"
        )
