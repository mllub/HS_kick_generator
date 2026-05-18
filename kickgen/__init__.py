"""kickgen — Parameterized DSP pipeline for procedural hardstyle kick generation."""

from .blocks import (
    Block,
    Gain,
    KickSynth,
    ParametricEQ,
    Waveshaper,
    Compressor,
    MultibandCompressor,
    Reverb,
    Limiter,
)
from .channel import Channel
from .pipeline import Pipeline
from .registry import BLOCK_REGISTRY

__all__ = [
    "Block",
    "Gain",
    "KickSynth",
    "ParametricEQ",
    "Waveshaper",
    "Compressor",
    "MultibandCompressor",
    "Reverb",
    "Limiter",
    "Channel",
    "Pipeline",
    "BLOCK_REGISTRY",
]

__version__ = "0.1.0"
