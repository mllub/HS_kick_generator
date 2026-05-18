# hardstyle-kick-rave

Train a generative model on hardstyle kick drums using [RAVE](https://github.com/acids-ircam/RAVE).

Configured for **44100 Hz, 0.4-second single-shot kick drums**.

---

## kickgen — Parameterized DSP Kick Synthesizer

`kickgen` is a Python library for procedurally generating hardstyle kick drum
samples using a parameterized DSP pipeline.  Every parameter is a named
float/int, making the library suitable for **Bayesian optimization** of kick
drum sounds.

### Install

```bash
pip install -e ".[audio]"
```

For GPU-accelerated loops (optional):

```bash
pip install -e ".[audio,fast]"
```

### Minimal example

```python
from kickgen import KickSynth, ParametricEQ, Gain, Channel, Pipeline

kick = KickSynth(start_freq=150, end_freq=40, sweep_time=0.5,
                 decay_ms=400, click_level=0.2)
eq   = ParametricEQ(n_bands=2)
gain = Gain(gain_db=-6)

ch = Channel(blocks=[("kick", kick), ("eq", eq), ("gain", gain)],
             pan=0.0, gain_db=0.0)

pipeline = Pipeline(channels=[("tail", ch)], master_gain_db=0.0)

audio = pipeline.render(length_seconds=2.0, sr=44100)  # (88200, 2) float32

import soundfile as sf
sf.write("kick.wav", audio, 44100)
```

### Getting and setting the flat parameter dict

```python
params = pipeline.get_params()
# {'tail.kick.start_freq': 150.0, 'tail.eq.band_0_freq': 1000.0, ...}

pipeline.set_params(**{"tail.kick.start_freq": 200.0, "tail.kick.end_freq": 50.0})
```

### Dump / load params via JSON (for the optimizer)

```python
import json

# Save
with open("params.json", "w") as f:
    json.dump(pipeline.get_params(), f, indent=2)

# Load and restore
with open("params.json") as f:
    saved = json.load(f)

pipeline.set_params(**saved)
```

### Parameter bounds (for Bayesian optimization)

```python
bounds = pipeline.param_bounds()
# {'tail.kick.start_freq': (20.0, 8000.0), ...}
```

### Running the full hardstyle example

```bash
python examples/basic_hardstyle.py
# Writes out.wav and prints the full flat param dict
```

### Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## RAVE training workflow

### Requirements
- Python 3.10
- CUDA-capable GPU (8 GB+ VRAM recommended)

### Install

```bash
conda create -n hs-kick python=3.10
conda activate hs-kick
pip install acids-rave --no-deps
pip install -r requirements.txt
```

### 1. Add kick samples

Drop your hardstyle kick WAV/MP3/FLAC files into `data/raw/`.

- 100 kicks minimum; 200–500 is ideal
- Avoid clipping, heavy silence, or mixed sub-genres

### 2. Preprocess

```bash
python scripts/preprocess.py
```

Per file:
- Resamples to 44100 Hz mono
- Detects the first transient onset and aligns to it (2 ms pre-roll)
- Trims or zero-pads to exactly 17640 samples (0.4 s)
- Peak-normalizes to −1 dBFS
- Saves 24-bit WAV to `data/processed/`
- Generates 21 augmented variants (EQ, clipping, polarity)

Then builds the LMDB training database at `data/kicks.mdb`.

### 3. Train

```bash
python scripts/train.py
```

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | 600000 | 400k–800k is good for 100–300 sample datasets |
| `--val-every` | 10000 | How often reconstructed audio is logged |
| `--name` | kick_rave | Subdirectory name under `outputs/` |
| `--gpu` | 0 | GPU index |

Monitor training:

```bash
tensorboard --logdir outputs/
```

---

## Project structure

```
hardstyle-kick-rave/
├── kickgen/               ← DSP synthesis library
│   ├── __init__.py
│   ├── blocks.py          ← Gain, KickSynth, ParametricEQ, Waveshaper, ...
│   ├── channel.py         ← Channel (ordered block chain)
│   └── pipeline.py        ← Pipeline (multi-channel stereo mix)
├── examples/
│   └── basic_hardstyle.py ← 4-channel hardstyle kick example
├── tests/
│   ├── test_blocks.py
│   ├── test_channel.py
│   └── test_pipeline.py
├── data/
│   ├── raw/               ← drop your kick samples here
│   ├── processed/         ← created by preprocess.py
│   └── kicks.mdb          ← LMDB training database
├── configs/
│   └── kick_rave.gin      ← model architecture config
├── scripts/
│   ├── preprocess.py      ← normalize / align / augment / build LMDB
│   └── train.py           ← launch training
├── pyproject.toml
└── outputs/
    └── kick_rave/         ← checkpoints and logs
```
