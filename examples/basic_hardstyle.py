"""Basic hardstyle kick example.

Builds a 4-channel Pipeline (tik, punch, tail, reverb) and renders 2 seconds
of audio to out.wav.  Prints the full flat parameter dict and total count.

Usage
-----
    python examples/basic_hardstyle.py
"""

from __future__ import annotations

import sys

import numpy as np

# Allow running from repo root without installing
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from kickgen import (
    Channel,
    Compressor,
    Gain,
    KickSynth,
    MultibandCompressor,
    ParametricEQ,
    Pipeline,
    Reverb,
    Waveshaper,
)

# ---------------------------------------------------------------------------
# tik — attack click, ~5 kHz
# ---------------------------------------------------------------------------
tik_kick = KickSynth(
    start_freq=5000, end_freq=3000, sweep_time=0.03, sweep_curve=3.0,
    length=2.0, attack_ms=0.5, decay_ms=80, click_level=0.8, click_decay_ms=3.0,
)
tik_eq = ParametricEQ(
    n_bands=2,
    **{
        "band_0_freq": 4000.0, "band_0_gain_db": 6.0, "band_0_Q": 0.7, "band_0_type": 2,  # high-shelf
        "band_1_freq": 500.0,  "band_1_gain_db": -3.0, "band_1_Q": 1.5, "band_1_type": 0,  # peak
    },
)
tik_gain = Gain(gain_db=-6)

tik_channel = Channel(
    blocks=[("kick", tik_kick), ("eq", tik_eq), ("gain", tik_gain)],
    pan=-0.0,
    gain_db=-3,
)

# ---------------------------------------------------------------------------
# punch — body, ~100 Hz
# ---------------------------------------------------------------------------
punch_kick = KickSynth(
    start_freq=120, end_freq=50, sweep_time=0.15, sweep_curve=1.5,
    length=2.0, attack_ms=3.0, decay_ms=200, click_level=0.1, click_decay_ms=5.0,
)
punch_eq = ParametricEQ(
    n_bands=3,
    **{
        "band_0_freq": 60.0,  "band_0_gain_db": 4.0,  "band_0_Q": 1.0, "band_0_type": 1,  # low-shelf
        "band_1_freq": 120.0, "band_1_gain_db": 3.0,  "band_1_Q": 2.0, "band_1_type": 0,  # peak
        "band_2_freq": 800.0, "band_2_gain_db": -6.0, "band_2_Q": 1.0, "band_2_type": 0,  # peak
    },
)
punch_comp = Compressor(
    threshold_db=-12, ratio=4, attack_ms=5, release_ms=80, knee_db=6, makeup_db=3,
)
punch_gain = Gain(gain_db=-3)

punch_channel = Channel(
    blocks=[
        ("kick", punch_kick),
        ("eq", punch_eq),
        ("compressor", punch_comp),
        ("gain", punch_gain),
    ],
    pan=-1,
    gain_db=0,
)

# ---------------------------------------------------------------------------
# tail — distorted pitched sub (the signature hardstyle sound)
# ---------------------------------------------------------------------------
tail_kick = KickSynth(
    start_freq=150, end_freq=40, sweep_time=0.6, sweep_curve=2.5,
    length=2.0, attack_ms=2.0, decay_ms=600, click_level=0.15, click_decay_ms=8.0,
)
tail_eq1 = ParametricEQ(
    n_bands=4,
    **{
        "band_0_freq": 80.0,   "band_0_gain_db": 6.0,  "band_0_Q": 1.0, "band_0_type": 1,  # low-shelf
        "band_1_freq": 120.0,  "band_1_gain_db": 8.0,  "band_1_Q": 1.5, "band_1_type": 0,  # peak
        "band_2_freq": 400.0,  "band_2_gain_db": -4.0, "band_2_Q": 1.0, "band_2_type": 0,  # peak
        "band_3_freq": 2000.0, "band_3_gain_db": 3.0,  "band_3_Q": 2.0, "band_3_type": 0,  # peak
    },
)
tail_ws = Waveshaper(drive=8.0, bias=0.05, mix=0.9, shape=0)
tail_eq2 = ParametricEQ(
    n_bands=4,
    **{
        "band_0_freq": 100.0,  "band_0_gain_db": -6.0, "band_0_Q": 0.8, "band_0_type": 0,  # peak
        "band_1_freq": 200.0,  "band_1_gain_db": 4.0,  "band_1_Q": 1.5, "band_1_type": 0,  # peak
        "band_2_freq": 1000.0, "band_2_gain_db": -3.0, "band_2_Q": 1.0, "band_2_type": 0,  # peak
        "band_3_freq": 4000.0, "band_3_gain_db": -4.0, "band_3_Q": 1.0, "band_3_type": 2,  # high-shelf
    },
)
tail_comp = Compressor(
    threshold_db=-18, ratio=6, attack_ms=2, release_ms=120, knee_db=4, makeup_db=8,
)
tail_mbc = MultibandCompressor(n_bands=3, **{"xover_0_hz": 200.0, "xover_1_hz": 2000.0})

tail_channel = Channel(
    blocks=[
        ("kick", tail_kick),
        ("eq1", tail_eq1),
        ("waveshaper", tail_ws),
        ("eq2", tail_eq2),
        ("compressor", tail_comp),
        ("mbc", tail_mbc),
    ],
    pan=-0,
    gain_db=0,
)

# ---------------------------------------------------------------------------
# reverb — air and space
# ---------------------------------------------------------------------------
reverb_kick = KickSynth(
    start_freq=100, end_freq=35, sweep_time=0.4, sweep_curve=2.0,
    length=2.0, attack_ms=5.0, decay_ms=500, click_level=0.0, click_decay_ms=5.0,
)
reverb_fx = Reverb(room_size=0.6, decay=0.65, damping=0.4, pre_delay_ms=15, mix=0.8)
reverb_gain = Gain(gain_db=-12)

reverb_channel = Channel(
    blocks=[("kick", reverb_kick), ("reverb", reverb_fx), ("gain", reverb_gain)],
    pan=0.0,
    gain_db=-6,
)

# ---------------------------------------------------------------------------
# Assemble Pipeline and render
# ---------------------------------------------------------------------------
pipeline = Pipeline(
    channels=[
        ("tik", tik_channel),
        ("punch", punch_channel),
        ("tail", tail_channel),
        ("reverb", reverb_channel),
    ],
    master_gain_db=0.0,
    use_limiter=True,
)

print("Rendering 2.0 seconds at 44100 Hz ...")
audio = pipeline.render(length_seconds=2.0, sr=44100)

# Write to file
output_path = "out.wav"
try:
    import soundfile as sf  # type: ignore[import]
    sf.write(output_path, audio, 44100, subtype="PCM_24")
    print(f"Saved to {output_path} (soundfile, PCM_24)")
except ImportError:
    from scipy.io import wavfile  # type: ignore[import]
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    wavfile.write(output_path, 44100, audio_int16)
    print(f"Saved to {output_path} (scipy wavfile, int16)")

# Print param dict
params = pipeline.get_params()
print(f"\nTotal parameters: {len(params)}")
print("\nFlat parameter dict:")
for k, v in sorted(params.items()):
    print(f"  {k}: {v}")
