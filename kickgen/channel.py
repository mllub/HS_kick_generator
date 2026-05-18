"""Channel: an ordered chain of Blocks with pan and gain."""

from __future__ import annotations

import numpy as np

from .blocks import Block


class Channel:
    """An ordered chain of DSP :class:`~kickgen.blocks.Block` objects.

    Processes blocks sequentially.  The first block is expected to be a source
    (e.g. :class:`~kickgen.blocks.KickSynth`) that generates audio from
    scratch; subsequent blocks transform it.

    Parameters
    ----------
    blocks:
        Ordered list of ``(name, block)`` pairs.
    pan:
        Stereo pan position in [-1, 1] (−1 = hard left, +1 = hard right).
    gain_db:
        Output gain in dB applied after block processing.
    """

    _PAN_BOUNDS: tuple[float, float] = (-1.0, 1.0)
    _GAIN_BOUNDS: tuple[float, float] = (-40.0, 12.0)

    def __init__(
        self,
        blocks: list[tuple[str, Block]],
        pan: float = 0.0,
        gain_db: float = 0.0,
    ) -> None:
        self.blocks: list[tuple[str, Block]] = list(blocks)
        self.pan = float(pan)
        self.gain_db = float(gain_db)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, sr: int, n_samples: int) -> np.ndarray:
        """Run the block chain and return a mono float32 array of length *n_samples*.

        The first block receives a zero array as its input (source blocks
        ignore it).  The gain_db is applied after all blocks.

        Parameters
        ----------
        sr:
            Sample rate in Hz.
        n_samples:
            Number of samples to render.
        """
        audio = np.zeros(n_samples, dtype=np.float32)
        for _name, block in self.blocks:
            audio = block.process(audio, sr)

        # Pad / truncate to n_samples
        if len(audio) < n_samples:
            audio = np.concatenate([audio, np.zeros(n_samples - len(audio), dtype=np.float32)])
        else:
            audio = audio[:n_samples]

        # Apply channel gain
        linear = 10.0 ** (self.gain_db / 20.0)
        return (audio * linear).astype(np.float32)

    # ------------------------------------------------------------------
    # Parameter interface
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, float]:
        """Return flat param dict with keys like ``"block_name.param_name"``."""
        params: dict[str, float] = {
            "pan": self.pan,
            "gain_db": self.gain_db,
        }
        for name, block in self.blocks:
            for k, v in block.get_params().items():
                params[f"{name}.{k}"] = v
        return params

    def set_params(self, **kwargs: float) -> None:
        """Set parameters by flat key ``"block_name.param_name"``."""
        for key, value in kwargs.items():
            if key == "pan":
                self.pan = float(value)
            elif key == "gain_db":
                self.gain_db = float(value)
            else:
                # Split on first '.' only
                dot = key.find(".")
                if dot < 0:
                    continue
                block_name = key[:dot]
                param_key = key[dot + 1:]
                for name, block in self.blocks:
                    if name == block_name:
                        block.set_params(**{param_key: value})
                        break

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for all parameters including pan and gain_db."""
        bounds: dict[str, tuple[float, float]] = {
            "pan": self._PAN_BOUNDS,
            "gain_db": self._GAIN_BOUNDS,
        }
        for name, block in self.blocks:
            for k, v in block.param_bounds().items():
                bounds[f"{name}.{k}"] = v
        return bounds

    def __repr__(self) -> str:
        block_names = [n for n, _ in self.blocks]
        return (
            f"Channel(blocks={block_names}, pan={self.pan}, gain_db={self.gain_db})"
        )
