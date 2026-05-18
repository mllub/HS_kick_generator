"""Registry mapping block type names to their classes."""

from kickgen.blocks import (
    Gain,
    KickSynth,
    ParametricEQ,
    Waveshaper,
    Compressor,
    MultibandCompressor,
    Reverb,
    Limiter,
)

BLOCK_REGISTRY: dict[str, type] = {
    "Gain": Gain,
    "KickSynth": KickSynth,
    "ParametricEQ": ParametricEQ,
    "Waveshaper": Waveshaper,
    "Compressor": Compressor,
    "MultibandCompressor": MultibandCompressor,
    "Reverb": Reverb,
    "Limiter": Limiter,
}
