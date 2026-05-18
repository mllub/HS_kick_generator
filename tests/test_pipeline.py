"""Tests for the Pipeline class."""

from __future__ import annotations

import numpy as np
import pytest

from kickgen.blocks import Compressor, Gain, KickSynth, ParametricEQ, Waveshaper
from kickgen.channel import Channel
from kickgen.pipeline import Pipeline

SR = 44100
DURATION = 0.5
N_SAMPLES = int(DURATION * SR)  # 22050


def _make_pipeline() -> Pipeline:
    """Build a minimal 2-channel pipeline for testing."""
    # Channel 1: simple synth + gain
    kick1 = KickSynth(length=DURATION, start_freq=120, end_freq=50)
    gain1 = Gain(gain_db=-6.0)
    ch1 = Channel(
        blocks=[("kick", kick1), ("gain", gain1)],
        pan=-0.3,
        gain_db=-3.0,
    )

    # Channel 2: synth + EQ + waveshaper
    kick2 = KickSynth(length=DURATION, start_freq=5000, end_freq=3000, decay_ms=80)
    eq2 = ParametricEQ(n_bands=2)
    ws2 = Waveshaper(drive=4.0, mix=0.7)
    ch2 = Channel(
        blocks=[("kick", kick2), ("eq", eq2), ("ws", ws2)],
        pan=0.3,
        gain_db=0.0,
    )

    return Pipeline(
        channels=[("low", ch1), ("high", ch2)],
        master_gain_db=0.0,
        use_limiter=True,
    )


class TestPipeline:
    def test_render_shape(self):
        pipeline = _make_pipeline()
        out = pipeline.render(DURATION, sr=SR)
        assert out.shape == (N_SAMPLES, 2), f"Expected ({N_SAMPLES}, 2), got {out.shape}"

    def test_render_dtype(self):
        pipeline = _make_pipeline()
        out = pipeline.render(DURATION, sr=SR)
        assert out.dtype == np.float32

    def test_render_finite(self):
        pipeline = _make_pipeline()
        out = pipeline.render(DURATION, sr=SR)
        assert np.all(np.isfinite(out)), "Output contains NaN or Inf"

    def test_render_range(self):
        """After limiter, output should stay within [-1.05, 1.05]."""
        pipeline = _make_pipeline()
        out = pipeline.render(DURATION, sr=SR)
        assert np.all(out >= -1.05), f"Min value: {out.min()}"
        assert np.all(out <= 1.05), f"Max value: {out.max()}"

    def test_render_not_silence(self):
        pipeline = _make_pipeline()
        out = pipeline.render(DURATION, sr=SR)
        assert np.any(out != 0.0)

    def test_get_params_returns_dict(self):
        pipeline = _make_pipeline()
        params = pipeline.get_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_get_params_keys_have_prefix(self):
        pipeline = _make_pipeline()
        params = pipeline.get_params()
        non_master_keys = [k for k in params if k not in ("master_gain_db",)
                           and not k.startswith("master_limiter")]
        for k in non_master_keys:
            assert "." in k, f"Expected dot-separated key, got: {k}"

    def test_set_params_roundtrip(self):
        pipeline = _make_pipeline()
        original = pipeline.get_params()
        # Set all params to their current values
        pipeline.set_params(**original)
        recovered = pipeline.get_params()
        for k, v in original.items():
            assert abs(recovered[k] - v) < 1e-9, f"Mismatch at {k}: {recovered[k]} vs {v}"

    def test_set_params_changes_value(self):
        pipeline = _make_pipeline()
        pipeline.set_params(**{"low.kick.start_freq": 300.0})
        params = pipeline.get_params()
        assert abs(params["low.kick.start_freq"] - 300.0) < 1e-9

    def test_param_bounds_covers_all_params(self):
        pipeline = _make_pipeline()
        params = pipeline.get_params()
        bounds = pipeline.param_bounds()
        for k in params:
            assert k in bounds, f"Missing bounds for: {k}"

    def test_master_limiter_params_accessible(self):
        pipeline = _make_pipeline()
        params = pipeline.get_params()
        assert "master_limiter.threshold_db" in params
        assert "master_limiter.release_ms" in params

    def test_master_gain_db_param(self):
        pipeline = _make_pipeline()
        pipeline.set_params(master_gain_db=-6.0)
        params = pipeline.get_params()
        assert abs(params["master_gain_db"] - (-6.0)) < 1e-9

    def test_no_limiter_option(self):
        """Pipeline without limiter should still render correctly."""
        kick = KickSynth(length=DURATION)
        ch = Channel(blocks=[("kick", kick)], pan=0.0, gain_db=0.0)
        pipeline = Pipeline(
            channels=[("ch", ch)],
            master_gain_db=0.0,
            use_limiter=False,
        )
        out = pipeline.render(DURATION, sr=SR)
        assert out.shape == (N_SAMPLES, 2)
        assert np.all(np.isfinite(out))

    def test_pan_applies_stereo_spread(self):
        """Hard-panned channels should differ between L and R."""
        kick = KickSynth(length=DURATION)
        ch_left = Channel(blocks=[("kick", kick)], pan=-1.0, gain_db=0.0)
        pipeline = Pipeline(
            channels=[("ch", ch_left)],
            master_gain_db=0.0,
            use_limiter=False,
        )
        out = pipeline.render(DURATION, sr=SR)
        # Left channel should have more energy than right
        left_energy = np.sum(out[:, 0] ** 2)
        right_energy = np.sum(out[:, 1] ** 2)
        assert left_energy > right_energy
