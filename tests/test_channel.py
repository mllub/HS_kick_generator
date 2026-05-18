"""Tests for the Channel class."""

from __future__ import annotations

import numpy as np
import pytest

from kickgen.blocks import Compressor, Gain, KickSynth, ParametricEQ
from kickgen.channel import Channel

SR = 44100


def _make_channel() -> Channel:
    kick = KickSynth(length=1.0)
    eq = ParametricEQ(n_bands=2)
    comp = Compressor()
    gain = Gain(gain_db=-6.0)
    return Channel(
        blocks=[
            ("kick", kick),
            ("eq", eq),
            ("compressor", comp),
            ("gain", gain),
        ],
        pan=0.1,
        gain_db=-3.0,
    )


class TestChannel:
    def test_process_runs(self):
        ch = _make_channel()
        out = ch.process(SR, SR)
        assert out.shape == (SR,)

    def test_process_output_finite(self):
        ch = _make_channel()
        out = ch.process(SR, SR)
        assert np.all(np.isfinite(out))

    def test_process_output_not_all_zeros(self):
        ch = _make_channel()
        out = ch.process(SR, SR)
        assert np.any(out != 0.0)

    def test_flat_param_namespace_set_get(self):
        ch = _make_channel()
        # Set a nested param via flat key
        ch.set_params(**{"eq.band_0_freq": 800.0})
        params = ch.get_params()
        assert abs(params["eq.band_0_freq"] - 800.0) < 1e-9

    def test_flat_param_multiple_blocks(self):
        ch = _make_channel()
        ch.set_params(**{
            "kick.start_freq": 200.0,
            "compressor.threshold_db": -30.0,
            "gain.gain_db": -12.0,
        })
        params = ch.get_params()
        assert abs(params["kick.start_freq"] - 200.0) < 1e-9
        assert abs(params["compressor.threshold_db"] - (-30.0)) < 1e-9
        assert abs(params["gain.gain_db"] - (-12.0)) < 1e-9

    def test_pan_and_gain_db_params(self):
        ch = _make_channel()
        ch.set_params(pan=0.5, gain_db=-8.0)
        params = ch.get_params()
        assert abs(params["pan"] - 0.5) < 1e-9
        assert abs(params["gain_db"] - (-8.0)) < 1e-9

    def test_param_bounds_includes_all_keys(self):
        ch = _make_channel()
        params = ch.get_params()
        bounds = ch.param_bounds()
        for k in params:
            assert k in bounds, f"Missing bounds for key: {k}"

    def test_param_roundtrip(self):
        ch = _make_channel()
        # Collect all params, modify them, set back, verify
        original = ch.get_params()
        modified = {k: v + 0.0 for k, v in original.items()}
        ch.set_params(**modified)
        recovered = ch.get_params()
        for k, v in original.items():
            assert abs(recovered[k] - v) < 1e-9, f"Mismatch at {k}: {recovered[k]} != {v}"

    def test_truncates_to_n_samples(self):
        # KickSynth with length=1.0 generates SR samples; request half
        kick = KickSynth(length=1.0)
        ch = Channel(blocks=[("kick", kick)], pan=0.0, gain_db=0.0)
        out = ch.process(SR, SR // 2)
        assert out.shape == (SR // 2,)

    def test_pads_to_n_samples(self):
        # KickSynth with length=0.1; request full second
        kick = KickSynth(length=0.1)
        ch = Channel(blocks=[("kick", kick)], pan=0.0, gain_db=0.0)
        out = ch.process(SR, SR)
        assert out.shape == (SR,)
