"""Tests for individual DSP blocks."""

from __future__ import annotations

import numpy as np
import pytest

from kickgen.blocks import (
    Compressor,
    Gain,
    KickSynth,
    Limiter,
    MultibandCompressor,
    ParametricEQ,
    Reverb,
    Waveshaper,
)

SR = 44100
N = SR  # 1 second of audio


def _white_noise(n: int = N, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32) * 0.5


# ---------------------------------------------------------------------------
# Gain
# ---------------------------------------------------------------------------

class TestGain:
    def test_output_shape(self):
        block = Gain(gain_db=0.0)
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = Gain(gain_db=-6.0)
        block.set_params(gain_db=-12.0)
        assert abs(block.get_params()["gain_db"] - (-12.0)) < 1e-9

    def test_output_finite(self):
        block = Gain(gain_db=6.0)
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))

    def test_gain_applied(self):
        block = Gain(gain_db=20.0)  # +20 dB = 10x linear
        audio = np.ones(100, dtype=np.float32) * 0.1
        out = block.process(audio, SR)
        np.testing.assert_allclose(out, audio * 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# KickSynth
# ---------------------------------------------------------------------------

class TestKickSynth:
    def test_output_shape(self):
        block = KickSynth(length=1.0)
        out = block.process(np.zeros(1, dtype=np.float32), SR)
        assert out.shape == (int(1.0 * SR),)

    def test_output_shape_custom_length(self):
        block = KickSynth(length=0.5)
        out = block.process(np.zeros(1, dtype=np.float32), SR)
        assert out.shape == (int(0.5 * SR),)

    def test_param_roundtrip(self):
        block = KickSynth()
        new_params = {
            "start_freq": 300.0,
            "end_freq": 50.0,
            "sweep_time": 0.4,
            "sweep_curve": 3.0,
            "length": 1.5,
            "attack_ms": 5.0,
            "decay_ms": 300.0,
            "click_level": 0.3,
            "click_decay_ms": 8.0,
        }
        block.set_params(**new_params)
        got = block.get_params()
        for k, v in new_params.items():
            assert abs(got[k] - v) < 1e-9, f"Param {k}: expected {v}, got {got[k]}"

    def test_output_finite(self):
        block = KickSynth()
        out = block.process(np.zeros(1, dtype=np.float32), SR)
        assert np.all(np.isfinite(out))

    def test_not_all_zeros(self):
        block = KickSynth()
        out = block.process(np.zeros(1, dtype=np.float32), SR)
        assert np.any(out != 0.0)

    def test_output_peak_normalized(self):
        block = KickSynth(click_level=0.0)
        out = block.process(np.zeros(1, dtype=np.float32), SR)
        assert abs(np.max(np.abs(out)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# ParametricEQ
# ---------------------------------------------------------------------------

class TestParametricEQ:
    def test_output_shape(self):
        block = ParametricEQ(n_bands=4)
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = ParametricEQ(n_bands=2)
        block.set_params(**{"band_0_freq": 500.0, "band_1_gain_db": -6.0})
        params = block.get_params()
        assert abs(params["band_0_freq"] - 500.0) < 1e-9
        assert abs(params["band_1_gain_db"] - (-6.0)) < 1e-9

    def test_output_finite(self):
        block = ParametricEQ(n_bands=3)
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))

    def test_zero_gain_passthrough(self):
        block = ParametricEQ(n_bands=2)  # all gain_db=0 by default
        audio = _white_noise()
        out = block.process(audio, SR)
        # With zero-gain peak EQ the signal should be very close to the input
        np.testing.assert_allclose(out, audio, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Waveshaper
# ---------------------------------------------------------------------------

class TestWaveshaper:
    def test_output_shape(self):
        block = Waveshaper()
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = Waveshaper()
        block.set_params(drive=10.0, bias=0.1, mix=0.5, shape=2.0)
        params = block.get_params()
        assert abs(params["drive"] - 10.0) < 1e-9
        assert abs(params["bias"] - 0.1) < 1e-9
        assert abs(params["mix"] - 0.5) < 1e-9
        assert abs(params["shape"] - 2.0) < 1e-9

    def test_output_finite(self):
        block = Waveshaper(drive=8.0, shape=0)
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("shape", [0, 1, 2, 3])
    def test_all_shapes_finite(self, shape):
        block = Waveshaper(shape=float(shape), drive=5.0)
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))

    def test_dry_wet(self):
        block = Waveshaper(drive=20.0, mix=0.0)  # fully dry
        audio = _white_noise()
        out = block.process(audio, SR)
        # mix=0 → output == dry input
        np.testing.assert_allclose(out, audio, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

class TestCompressor:
    def test_output_shape(self):
        block = Compressor()
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = Compressor()
        new = dict(threshold_db=-24.0, ratio=8.0, attack_ms=20.0,
                   release_ms=200.0, knee_db=3.0, makeup_db=12.0)
        block.set_params(**new)
        params = block.get_params()
        for k, v in new.items():
            assert abs(params[k] - v) < 1e-9

    def test_output_finite(self):
        block = Compressor()
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))

    def test_reduces_dynamics(self):
        """Compressor should reduce the peak-to-mean ratio."""
        audio = _white_noise()
        block = Compressor(threshold_db=-20, ratio=10, attack_ms=1, release_ms=50,
                           makeup_db=0)
        out = block.process(audio, SR)
        peak_in = np.max(np.abs(audio))
        peak_out = np.max(np.abs(out))
        # With ratio=10 and no makeup gain, output peak should be reduced
        assert peak_out <= peak_in + 1e-4


# ---------------------------------------------------------------------------
# MultibandCompressor
# ---------------------------------------------------------------------------

class TestMultibandCompressor:
    def test_output_shape(self):
        block = MultibandCompressor(n_bands=3)
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = MultibandCompressor(n_bands=3)
        block.set_params(**{"xover_0_hz": 250.0, "band_1_threshold_db": -24.0})
        params = block.get_params()
        assert abs(params["xover_0_hz"] - 250.0) < 1e-9
        assert abs(params["band_1_threshold_db"] - (-24.0)) < 1e-9

    def test_output_finite(self):
        block = MultibandCompressor(n_bands=3)
        out = block.process(_white_noise(), SR)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Reverb
# ---------------------------------------------------------------------------

class TestReverb:
    def test_output_shape(self):
        block = Reverb()
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = Reverb()
        new = dict(room_size=0.8, decay=0.7, damping=0.5, pre_delay_ms=20.0, mix=0.4)
        block.set_params(**new)
        params = block.get_params()
        for k, v in new.items():
            assert abs(params[k] - v) < 1e-9

    def test_output_finite(self):
        block = Reverb(room_size=0.9, decay=0.9, damping=0.5)
        out = block.process(_white_noise(N // 4), SR)
        assert np.all(np.isfinite(out))

    def test_dry_passthrough(self):
        block = Reverb(mix=0.0)
        audio = _white_noise()
        out = block.process(audio, SR)
        np.testing.assert_allclose(out, audio, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

class TestLimiter:
    def test_output_shape(self):
        block = Limiter()
        audio = _white_noise()
        out = block.process(audio, SR)
        assert out.shape == audio.shape

    def test_param_roundtrip(self):
        block = Limiter()
        block.set_params(threshold_db=-3.0, release_ms=100.0)
        params = block.get_params()
        assert abs(params["threshold_db"] - (-3.0)) < 1e-9
        assert abs(params["release_ms"] - 100.0) < 1e-9

    def test_output_finite(self):
        block = Limiter()
        # Feed loud signal
        audio = _white_noise() * 5.0
        out = block.process(audio, SR)
        assert np.all(np.isfinite(out))

    def test_output_range(self):
        """After Limiter, output should stay in [-1.05, 1.05]."""
        block = Limiter(threshold_db=-1.0)
        audio = _white_noise() * 10.0  # very loud
        out = block.process(audio, SR)
        assert np.all(out >= -1.05), f"min={out.min()}"
        assert np.all(out <= 1.05), f"max={out.max()}"
