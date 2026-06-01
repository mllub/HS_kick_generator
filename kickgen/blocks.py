"""DSP building blocks for hardstyle kick drum synthesis.

All blocks expose named float/int parameters suitable for Bayesian optimization.
"""

from __future__ import annotations

import abc
import math
from typing import Any

import numpy as np
import scipy.signal as sig

# ---------------------------------------------------------------------------
# Optional numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit  # type: ignore[import]
    _NUMBA = True
except ImportError:
    def _njit(f):  # type: ignore[misc]
        return f
    _NUMBA = False


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Block(abc.ABC):
    """Abstract base for all DSP pipeline blocks.

    Every parameter must be accessible via :meth:`get_params` /
    :meth:`set_params` so that an external optimizer can treat the full
    pipeline as a single flat vector.
    """

    @abc.abstractmethod
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process *audio* (mono float32, shape ``(N,)``) and return same shape.

        Source blocks (e.g. :class:`KickSynth`) ignore *audio* and generate
        from scratch; they still accept it for pipeline compatibility.
        """

    @abc.abstractmethod
    def get_params(self) -> dict[str, float]:
        """Return a dict of all current parameter values."""

    @abc.abstractmethod
    def set_params(self, **kwargs: float) -> None:
        """Set one or more parameters by name."""

    @abc.abstractmethod
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """Return ``{name: (low, high)}`` for every parameter."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# Gain
# ---------------------------------------------------------------------------

class Gain(Block):
    """Simple gain stage (dB).

    Parameters
    ----------
    gain_db:
        Gain in decibels. Default 0.0.
    """

    _BOUNDS: dict[str, tuple[float, float]] = {"gain_db": (-60.0, 24.0)}

    def __init__(self, gain_db: float = 0.0) -> None:
        self.gain_db = float(gain_db)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        linear = 10.0 ** (self.gain_db / 20.0)
        return (audio * linear).astype(np.float32)

    def get_params(self) -> dict[str, float]:
        return {"gain_db": self.gain_db}

    def set_params(self, **kwargs: float) -> None:
        if "gain_db" in kwargs:
            self.gain_db = float(kwargs["gain_db"])

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._BOUNDS)


# ---------------------------------------------------------------------------
# KickSynth
# ---------------------------------------------------------------------------

N_ENV_PTS = 4   # envelope points for both pitch and amplitude envelopes
N_HARMONICS = 6  # harmonics above the fundamental (2nd … 7th)


class KickSynth(Block):
    """Synthesize a kick drum from scratch (ignores audio input).

    Generates a pitched sine with up to six harmonics.  Both the pitch and
    amplitude envelopes are piecewise-linear, each defined by N_ENV_PTS
    (time, value) knot points that are sorted internally before use.

    Pitch envelope
    --------------
    Each knot ``pitch_pt_N_v`` is a normalised value in [0, 1].  The actual
    frequency at that point is ``pitch_pt_N_v * freq_scale``.

    Amplitude envelope
    ------------------
    Each knot ``amp_pt_N_v`` is an amplitude in [0, 1].

    Harmonics
    ---------
    ``harm_N_amp`` and ``harm_N_phase`` (degrees) for harmonics 2–7 above the
    fundamental.  All amplitudes default to 0 (harmonics silent).
    """

    _N_ENV_PTS = N_ENV_PTS
    _N_HARMONICS = N_HARMONICS

    # Default knot points — (time_s, value)
    _DEFAULT_PITCH: list[list[float]] = [
        [0.000, 1.00],
        [0.080, 0.35],
        [0.300, 0.20],
        [1.000, 0.18],
    ]
    _DEFAULT_AMP: list[list[float]] = [
        [0.000, 0.00],
        [0.004, 1.00],
        [0.200, 0.60],
        [1.000, 0.00],
    ]

    def __init__(
        self,
        freq_scale: float = 150.0,
        length: float = 1.0,
        pitch_pts: list[list[float]] | None = None,
        amp_pts: list[list[float]] | None = None,
        harmonics: list[list[float]] | None = None,
    ) -> None:
        self.freq_scale = float(freq_scale)
        self.length = float(length)

        # Deep-copy defaults so instances don't share mutable state
        import copy
        self._pitch_pts: list[list[float]] = (
            copy.deepcopy(pitch_pts) if pitch_pts is not None
            else copy.deepcopy(self._DEFAULT_PITCH)
        )
        self._amp_pts: list[list[float]] = (
            copy.deepcopy(amp_pts) if amp_pts is not None
            else copy.deepcopy(self._DEFAULT_AMP)
        )
        # [[amp, phase_deg], ...]
        self._harmonics: list[list[float]] = (
            copy.deepcopy(harmonics) if harmonics is not None
            else [[0.0, 0.0] for _ in range(self._N_HARMONICS)]
        )

    # ------------------------------------------------------------------
    # Internal helpers exposed for the GUI
    # ------------------------------------------------------------------

    def sorted_pitch_pts(self) -> list[tuple[float, float]]:
        """Return pitch knots sorted by time as (t, v) tuples."""
        return sorted((p[0], p[1]) for p in self._pitch_pts)

    def sorted_amp_pts(self) -> list[tuple[float, float]]:
        """Return amplitude knots sorted by time as (t, v) tuples."""
        return sorted((p[0], p[1]) for p in self._amp_pts)

    # ------------------------------------------------------------------
    # DSP
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ARG002
        n = int(self.length * sr)
        t = np.arange(n, dtype=np.float64) / sr

        # Pin endpoint times so they are always valid regardless of how params
        # were set (e.g. by the optimizer or deserialized from an old file).
        self._pitch_pts[0][0] = 0.0
        self._pitch_pts[-1][0] = self.length
        self._amp_pts[0][0] = 0.0
        self._amp_pts[-1][0] = self.length

        # --- Pitch envelope ---
        pts = self.sorted_pitch_pts()
        p_times = [p[0] for p in pts]
        p_vals  = [p[1] for p in pts]
        freq = np.interp(t, p_times, p_vals) * self.freq_scale
        freq = np.maximum(freq, 0.5)   # guard against 0 Hz

        # --- Phase via integration ---
        phase = np.cumsum(2.0 * np.pi * freq / sr)

        # --- Fundamental + harmonics ---
        body = np.sin(phase)
        for i, (h_amp, h_phase_deg) in enumerate(self._harmonics):
            if h_amp > 1e-6:
                harmonic_num = i + 2   # 2nd, 3rd, … 7th harmonic
                h_phase_rad = math.radians(h_phase_deg)
                body = body + h_amp * np.sin(harmonic_num * phase + h_phase_rad)

        # --- Amplitude envelope ---
        apts = self.sorted_amp_pts()
        a_times = [p[0] for p in apts]
        a_vals  = [p[1] for p in apts]
        amp_env = np.interp(t, a_times, a_vals)
        body = body * amp_env

        # --- Peak-normalise ---
        peak = np.max(np.abs(body))
        if peak > 1e-9:
            body = body / peak

        return body.astype(np.float32)

    # ------------------------------------------------------------------
    # Param interface
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, float]:
        params: dict[str, float] = {
            "freq_scale": self.freq_scale,
            "length": self.length,
        }
        for i, (t, v) in enumerate(self._pitch_pts):
            params[f"pitch_pt_{i}_t"] = t
            params[f"pitch_pt_{i}_v"] = v
        for i, (t, v) in enumerate(self._amp_pts):
            params[f"amp_pt_{i}_t"] = t
            params[f"amp_pt_{i}_v"] = v
        for i, (amp, phase) in enumerate(self._harmonics):
            params[f"harm_{i}_amp"]   = amp
            params[f"harm_{i}_phase"] = phase
        return params

    def set_params(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if k == "freq_scale":
                self.freq_scale = float(v)
            elif k == "length":
                self.length = float(v)
            elif k.startswith("pitch_pt_"):
                # pitch_pt_N_t  or  pitch_pt_N_v
                parts = k.split("_")   # ["pitch","pt","N","t/v"]
                idx, field = int(parts[2]), parts[3]
                if 0 <= idx < len(self._pitch_pts):
                    self._pitch_pts[idx][0 if field == "t" else 1] = float(v)
            elif k.startswith("amp_pt_"):
                parts = k.split("_")   # ["amp","pt","N","t/v"]
                idx, field = int(parts[2]), parts[3]
                if 0 <= idx < len(self._amp_pts):
                    self._amp_pts[idx][0 if field == "t" else 1] = float(v)
            elif k.startswith("harm_"):
                parts = k.split("_")   # ["harm","N","amp/phase"]
                idx, field = int(parts[1]), parts[2]
                if 0 <= idx < len(self._harmonics):
                    self._harmonics[idx][0 if field == "amp" else 1] = float(v)

    def param_bounds(self) -> dict[str, Any]:
        bounds: dict[str, Any] = {
            "freq_scale": (20.0, 1000.0),
            "length":     (0.1,  4.0),
        }
        for i in range(self._N_ENV_PTS):
            bounds[f"pitch_pt_{i}_t"] = (0.0, 4.0)
            bounds[f"pitch_pt_{i}_v"] = (0.0, 1.0)
        for i in range(self._N_ENV_PTS):
            bounds[f"amp_pt_{i}_t"] = (0.0, 4.0)
            bounds[f"amp_pt_{i}_v"] = (0.0, 1.0)
        for i in range(self._N_HARMONICS):
            bounds[f"harm_{i}_amp"]   = (0.0,    1.0)
            bounds[f"harm_{i}_phase"] = (-180.0, 180.0)
        return bounds


# ---------------------------------------------------------------------------
# ParametricEQ
# ---------------------------------------------------------------------------

def _design_biquad(
    band_type: int,
    freq: float,
    gain_db: float,
    q: float,
    sr: int,
) -> np.ndarray:
    """Return a single biquad as a (1, 6) SOS row.

    Uses Audio EQ Cookbook formulas.
    band_type: 0=peak, 1=low_shelf, 2=high_shelf
    """
    w0 = 2.0 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    A = 10.0 ** (gain_db / 40.0)

    if band_type == 1:  # low shelf, S=1 → alpha = sin(w0)/2 * sqrt(2)
        alpha = sin_w0 / 2.0 * np.sqrt(2.0)
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    elif band_type == 2:  # high shelf, S=1 → alpha = sin(w0)/2 * sqrt(2)
        alpha = sin_w0 / 2.0 * np.sqrt(2.0)
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:  # peak (band_type == 0)
        alpha = sin_w0 / (2.0 * q)
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

    # Normalize by a0 and return as SOS [b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


class ParametricEQ(Block):
    """N-band parametric equalizer using biquad filters.

    Parameters are named ``band_N_freq``, ``band_N_gain_db``, ``band_N_Q``,
    ``band_N_type`` for band index N (0-based).

    Parameters
    ----------
    n_bands:
        Number of EQ bands.
    """

    def __init__(self, n_bands: int = 4, **band_kwargs: float) -> None:
        self.n_bands = n_bands
        # Initialize defaults
        self._freq: list[float] = [1000.0] * n_bands
        self._gain_db: list[float] = [0.0] * n_bands
        self._q: list[float] = [1.0] * n_bands
        self._type: list[float] = [0.0] * n_bands  # stored as float for optimizer

        # Apply any kwargs
        for k, v in band_kwargs.items():
            self._set_band_param(k, float(v))

    def _set_band_param(self, key: str, value: float) -> None:
        # key format: band_N_freq / band_N_gain_db / band_N_Q / band_N_type
        parts = key.split("_")
        if len(parts) < 3 or parts[0] != "band":
            return
        try:
            n = int(parts[1])
        except ValueError:
            return
        if n < 0 or n >= self.n_bands:
            return
        attr = "_".join(parts[2:])
        if attr == "freq":
            self._freq[n] = value
        elif attr == "gain_db":
            self._gain_db[n] = value
        elif attr == "Q":
            self._q[n] = value
        elif attr == "type":
            self._type[n] = value

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        out = audio.astype(np.float64)
        for n in range(self.n_bands):
            freq = float(np.clip(self._freq[n], 20.0, sr / 2.0 - 1.0))
            q = float(np.clip(self._q[n], 0.01, 100.0))
            sos = _design_biquad(int(self._type[n]), freq, self._gain_db[n], q, sr)
            out = sig.sosfilt(sos, out)
        return out.astype(np.float32)

    def get_params(self) -> dict[str, float]:
        params: dict[str, float] = {}
        for n in range(self.n_bands):
            params[f"band_{n}_freq"] = self._freq[n]
            params[f"band_{n}_gain_db"] = self._gain_db[n]
            params[f"band_{n}_Q"] = self._q[n]
            params[f"band_{n}_type"] = self._type[n]
        return params

    def set_params(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self._set_band_param(k, float(v))

    def param_bounds(self) -> dict:
        bounds: dict = {}
        for n in range(self.n_bands):
            bounds[f"band_{n}_freq"] = (20.0, 20000.0)
            bounds[f"band_{n}_gain_db"] = (-40.0, 40.0)
            bounds[f"band_{n}_Q"] = (0.1, 10.0)
            bounds[f"band_{n}_type"] = ["peak", "low_shelf", "high_shelf"]
        return bounds


# ---------------------------------------------------------------------------
# Waveshaper
# ---------------------------------------------------------------------------

class Waveshaper(Block):
    """Nonlinear waveshaper with selectable transfer function.

    Parameters
    ----------
    drive:
        Pre-gain multiplier before shaping.
    bias:
        DC offset added before shaping.
    mix:
        Dry/wet mix (0 = dry, 1 = fully shaped).
    shape:
        0 = tanh, 1 = soft_clip (x/(1+|x|)), 2 = hard_clip, 3 = asymmetric tanh.
    """

    _BOUNDS: dict = {
        "drive": (0.1, 40.0),
        "bias": (-0.5, 0.5),
        "mix": (0.0, 1.0),
        "shape": ["tanh", "soft_clip", "hard_clip", "asymmetric"],
    }

    def __init__(
        self,
        drive: float = 4.0,
        bias: float = 0.0,
        mix: float = 1.0,
        shape: float = 0.0,
    ) -> None:
        self.drive = float(drive)
        self.bias = float(bias)
        self.mix = float(mix)
        self.shape = float(shape)

    def _shape_fn(self, x: np.ndarray) -> np.ndarray:
        s = int(np.clip(self.shape, 0, 3))
        if s == 0:
            return np.tanh(x * self.drive)
        elif s == 1:
            driven = x * self.drive
            return driven / (1.0 + np.abs(driven))
        elif s == 2:
            return np.clip(x * self.drive, -1.0, 1.0)
        else:  # asymmetric
            out = np.where(
                x > 0,
                np.tanh(x * self.drive),
                np.tanh(x * self.drive * 0.5),
            )
            return out

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:  # noqa: ARG002
        dry = audio.astype(np.float64)
        biased = dry + self.bias
        shaped = self._shape_fn(biased)
        # Normalize shaped to peak 1.0 before mixing
        peak = np.max(np.abs(shaped))
        if peak > 1e-9:
            shaped = shaped / peak
        out = self.mix * shaped + (1.0 - self.mix) * dry
        return out.astype(np.float32)

    def get_params(self) -> dict[str, float]:
        return {
            "drive": self.drive,
            "bias": self.bias,
            "mix": self.mix,
            "shape": self.shape,
        }

    def set_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                # "shape" may arrive as an int index from the GUI dropdown
                setattr(self, k, float(v))

    def param_bounds(self) -> dict:
        return dict(self._BOUNDS)


# ---------------------------------------------------------------------------
# Compressor (internal loop, optionally JIT-compiled)
# ---------------------------------------------------------------------------

@_njit
def _compress_loop(
    audio: np.ndarray,
    attack_coef: float,
    release_coef: float,
    threshold_lin: float,
    ratio: float,
    knee_width_lin: float,
    makeup_lin: float,
) -> np.ndarray:
    """Sample-by-sample compressor envelope follower and gain computer.

    Uses a soft-knee gain computer with feed-forward RMS approximation.
    """
    n = len(audio)
    out = np.empty(n, dtype=np.float32)
    env = 0.0
    half_knee = knee_width_lin / 2.0

    for i in range(n):
        x = audio[i]
        level = abs(x)

        # Envelope follower
        if level > env:
            env = attack_coef * env + (1.0 - attack_coef) * level
        else:
            env = release_coef * env + (1.0 - release_coef) * level

        # Gain computer (soft knee in linear domain approximation)
        if env <= (threshold_lin - half_knee):
            cs = 1.0
        elif env >= (threshold_lin + half_knee):
            # Hard compression region
            if env > 1e-30:
                cs = (threshold_lin * (env / threshold_lin) ** (1.0 / ratio)) / env
            else:
                cs = 1.0
        else:
            # Soft knee interpolation
            if half_knee > 1e-30:
                t = (env - (threshold_lin - half_knee)) / (2.0 * half_knee)
            else:
                t = 1.0
            above = threshold_lin + half_knee
            if above > 1e-30 and env > 1e-30:
                cs_hard = (threshold_lin * (above / threshold_lin) ** (1.0 / ratio)) / above
            else:
                cs_hard = 1.0
            cs = 1.0 + t * (cs_hard - 1.0)

        out[i] = np.float32(x * cs * makeup_lin)

    return out


class Compressor(Block):
    """Feed-forward dynamics compressor with soft knee.

    Parameters
    ----------
    threshold_db:
        Compression threshold in dBFS.
    ratio:
        Compression ratio (N:1).
    attack_ms:
        Attack time in milliseconds.
    release_ms:
        Release time in milliseconds.
    knee_db:
        Soft-knee width in dB.
    makeup_db:
        Make-up gain applied after compression.
    """

    _BOUNDS: dict[str, tuple[float, float]] = {
        "threshold_db": (-60.0, 0.0),
        "ratio": (1.0, 40.0),
        "attack_ms": (0.01, 200.0),
        "release_ms": (1.0, 2000.0),
        "knee_db": (0.0, 24.0),
        "makeup_db": (-12.0, 24.0),
    }

    def __init__(
        self,
        threshold_db: float = -18.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        knee_db: float = 6.0,
        makeup_db: float = 6.0,
    ) -> None:
        self.threshold_db = float(threshold_db)
        self.ratio = float(ratio)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.knee_db = float(knee_db)
        self.makeup_db = float(makeup_db)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        attack_coef = math.exp(-1.0 / (self.attack_ms * sr / 1000.0))
        release_coef = math.exp(-1.0 / (self.release_ms * sr / 1000.0))
        threshold_lin = 10.0 ** (self.threshold_db / 20.0)
        knee_width_lin = 10.0 ** (self.knee_db / 20.0) - 1.0  # approx linear knee
        makeup_lin = 10.0 ** (self.makeup_db / 20.0)

        x = audio.astype(np.float32)
        return _compress_loop(
            x, attack_coef, release_coef,
            threshold_lin, self.ratio, knee_width_lin, makeup_lin,
        )

    def get_params(self) -> dict[str, float]:
        return {
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "knee_db": self.knee_db,
            "makeup_db": self.makeup_db,
        }

    def set_params(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._BOUNDS)


# ---------------------------------------------------------------------------
# MultibandCompressor
# ---------------------------------------------------------------------------

def _lr4_sos(cutoff_hz: float, sr: int, btype: str) -> np.ndarray:
    """4th-order Linkwitz-Riley crossover filter as SOS (2x 2nd-order Butterworth)."""
    nyq = sr / 2.0
    wn = float(np.clip(cutoff_hz, 10.0, nyq - 1.0)) / nyq
    sos2 = sig.butter(2, wn, btype=btype, output="sos")
    # Apply twice for LR4 (cascade two 2nd-order BW)
    sos4 = np.vstack([sos2, sos2])
    return sos4


class MultibandCompressor(Block):
    """Multiband compressor with Linkwitz-Riley crossovers.

    Parameters
    ----------
    n_bands:
        Number of frequency bands (default 3).
    """

    def __init__(self, n_bands: int = 3, **kwargs: float) -> None:
        self.n_bands = n_bands
        n_xover = n_bands - 1

        # Default crossover frequencies
        default_xovers = [200.0, 2000.0, 8000.0, 16000.0]
        self._xover: list[float] = [
            float(default_xovers[i]) for i in range(n_xover)
        ]

        # Per-band compressors
        self._compressors: list[Compressor] = [
            Compressor() for _ in range(n_bands)
        ]

        # Apply any kwargs
        for k, v in kwargs.items():
            self._dispatch_set(k, float(v))

    def _dispatch_set(self, key: str, value: float) -> None:
        parts = key.split("_", 2)
        if key.startswith("xover_") and len(parts) >= 3 and parts[2] == "hz":
            try:
                n = int(parts[1])
                if 0 <= n < len(self._xover):
                    self._xover[n] = value
            except ValueError:
                pass
        elif key.startswith("band_"):
            # band_N_param_name
            rest = key[len("band_"):]
            idx_end = rest.find("_")
            if idx_end < 0:
                return
            try:
                n = int(rest[:idx_end])
            except ValueError:
                return
            param = rest[idx_end + 1:]
            if 0 <= n < self.n_bands:
                self._compressors[n].set_params(**{param: value})

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Build band signals via LR4 crossovers
        bands: list[np.ndarray] = []
        remainder = audio.astype(np.float64)

        for i, xf in enumerate(self._xover):
            lp_sos = _lr4_sos(xf, sr, "low")
            hp_sos = _lr4_sos(xf, sr, "high")
            low = sig.sosfilt(lp_sos, remainder).astype(np.float32)
            remainder = sig.sosfilt(hp_sos, remainder)
            bands.append(low)

        bands.append(remainder.astype(np.float32))

        # Compress each band
        output = np.zeros(len(audio), dtype=np.float32)
        for n, (band, comp) in enumerate(zip(bands, self._compressors)):
            output += comp.process(band, sr)

        return output

    def get_params(self) -> dict[str, float]:
        params: dict[str, float] = {}
        for i, xf in enumerate(self._xover):
            params[f"xover_{i}_hz"] = xf
        for n, comp in enumerate(self._compressors):
            for k, v in comp.get_params().items():
                params[f"band_{n}_{k}"] = v
        return params

    def set_params(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self._dispatch_set(k, float(v))

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        bounds: dict[str, tuple[float, float]] = {}
        for i in range(len(self._xover)):
            bounds[f"xover_{i}_hz"] = (20.0, 20000.0)
        for n in range(self.n_bands):
            for k, (lo, hi) in self._compressors[n].param_bounds().items():
                bounds[f"band_{n}_{k}"] = (lo, hi)
        return bounds


# ---------------------------------------------------------------------------
# Reverb (Freeverb-style)
# ---------------------------------------------------------------------------

# Freeverb standard delay lengths at 44100 Hz (Jezar at Dreampoint)
_COMB_DELAYS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
_ALLPASS_DELAYS = [556, 441, 341, 225]
_ALLPASS_COEF = 0.5


@_njit
def _comb_filter(
    audio: np.ndarray,
    delay: int,
    feedback: float,
    damping: float,
) -> np.ndarray:
    """Single comb filter with one-pole lowpass in the feedback path."""
    n = len(audio)
    buf = np.zeros(delay, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    lp = 0.0
    idx = 0
    for i in range(n):
        delayed = buf[idx]
        lp = lp * damping + delayed * (1.0 - damping)
        buf[idx] = audio[i] + feedback * lp
        out[i] = delayed
        idx = (idx + 1) % delay
    return out


@_njit
def _allpass_filter(audio: np.ndarray, delay: int, coef: float) -> np.ndarray:
    """Single allpass filter."""
    n = len(audio)
    buf = np.zeros(delay, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    idx = 0
    for i in range(n):
        delayed = buf[idx]
        v = audio[i] - coef * delayed
        buf[idx] = v
        out[i] = delayed + coef * v
        idx = (idx + 1) % delay
    return out


class Reverb(Block):
    """Freeverb-style reverb: 8 parallel combs + 4 series allpass filters.

    Uses the Freeverb algorithm (Jezar at Dreampoint) — 8 comb filters with
    prime-spaced delay lengths give a smooth, metallic-free spectral response,
    followed by 4 allpass stages for diffusion.

    Parameters
    ----------
    room_size:
        Scales the comb/allpass delay lengths (0.1–1.0).
    decay:
        Overall reverb decay (affects comb feedback gain).
    damping:
        High-frequency damping in comb feedback (0 = bright, 1 = dark).
    pre_delay_ms:
        Pre-delay in milliseconds.
    mix:
        Dry/wet mix (0 = dry, 1 = full wet).
    """

    _BOUNDS: dict[str, tuple[float, float]] = {
        "room_size": (0.1, 1.0),
        "decay": (0.1, 0.99),
        "damping": (0.0, 0.99),
        "pre_delay_ms": (0.0, 100.0),
        "mix": (0.0, 1.0),
    }

    def __init__(
        self,
        room_size: float = 0.5,
        decay: float = 0.5,
        damping: float = 0.3,
        pre_delay_ms: float = 10.0,
        mix: float = 0.25,
    ) -> None:
        self.room_size = float(room_size)
        self.decay = float(decay)
        self.damping = float(damping)
        self.pre_delay_ms = float(pre_delay_ms)
        self.mix = float(mix)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        dry = audio.astype(np.float64)

        # Pre-delay
        pre_samples = int(self.pre_delay_ms * sr / 1000.0)
        if pre_samples > 0:
            delayed_input = np.concatenate([np.zeros(pre_samples), dry[:-pre_samples]])
        else:
            delayed_input = dry

        scale = self.room_size * sr / 44100.0
        feedback = 0.9 * self.decay

        # 8 parallel comb filters (Freeverb delay lengths)
        wet = np.zeros_like(dry)
        for base_delay in _COMB_DELAYS:
            d = max(1, int(base_delay * scale))
            wet += _comb_filter(delayed_input, d, feedback, self.damping)
        wet *= 1.0 / len(_COMB_DELAYS)

        # 4 series allpass filters for diffusion
        for base_delay in _ALLPASS_DELAYS:
            d = max(1, int(base_delay * scale))
            wet = _allpass_filter(wet, d, _ALLPASS_COEF)

        out = (1.0 - self.mix) * dry + self.mix * wet
        return out.astype(np.float32)

    def get_params(self) -> dict[str, float]:
        return {
            "room_size": self.room_size,
            "decay": self.decay,
            "damping": self.damping,
            "pre_delay_ms": self.pre_delay_ms,
            "mix": self.mix,
        }

    def set_params(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, float(v))

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._BOUNDS)


# ---------------------------------------------------------------------------
# Limiter
# ---------------------------------------------------------------------------

class Limiter(Block):
    """Fast-attack limiter for master bus protection.

    Essentially a :class:`Compressor` with fixed high ratio and fast attack,
    plus hard-clipping of any residual peaks.

    Parameters
    ----------
    threshold_db:
        Limiter threshold in dBFS.
    release_ms:
        Release time in milliseconds.
    """

    _BOUNDS: dict[str, tuple[float, float]] = {
        "threshold_db": (-12.0, 0.0),
        "release_ms": (1.0, 500.0),
    }

    def __init__(
        self,
        threshold_db: float = -1.0,
        release_ms: float = 50.0,
    ) -> None:
        self.threshold_db = float(threshold_db)
        self.release_ms = float(release_ms)
        self._comp = Compressor(
            threshold_db=threshold_db,
            ratio=100.0,
            attack_ms=0.1,
            release_ms=release_ms,
            knee_db=0.5,
            makeup_db=0.0,
        )

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self._comp.threshold_db = self.threshold_db
        self._comp.release_ms = self.release_ms
        compressed = self._comp.process(audio, sr)
        # Hard-clip any remaining peaks
        return np.clip(compressed, -1.0, 1.0).astype(np.float32)

    def get_params(self) -> dict[str, float]:
        return {
            "threshold_db": self.threshold_db,
            "release_ms": self.release_ms,
        }

    def set_params(self, **kwargs: float) -> None:
        if "threshold_db" in kwargs:
            self.threshold_db = float(kwargs["threshold_db"])
        if "release_ms" in kwargs:
            self.release_ms = float(kwargs["release_ms"])

    def param_bounds(self) -> dict[str, tuple[float, float]]:
        return dict(self._BOUNDS)
