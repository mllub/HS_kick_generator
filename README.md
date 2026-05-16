# hardstyle-kick-rave

Train a generative model on hardstyle kick drums using [RAVE](https://github.com/acids-ircam/RAVE).

Configured for **44100 Hz, 0.4-second single-shot kick drums**.

---

## Setup

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

---

## Workflow

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
├── data/
│   ├── raw/               ← drop your kick samples here
│   ├── processed/         ← created by preprocess.py
│   └── kicks.mdb          ← LMDB training database
├── configs/
│   └── kick_rave.gin      ← model architecture config
├── scripts/
│   ├── preprocess.py      ← normalize / align / augment / build LMDB
│   └── train.py           ← launch training
└── outputs/
    └── kick_rave/         ← checkpoints and logs
```
