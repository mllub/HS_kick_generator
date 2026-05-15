# hardstyle-kick-rave

Generate hardstyle kick drums using [RAVE](https://github.com/acids-ircam/RAVE) — a variational autoencoder for fast, high-quality audio synthesis.

Configured for **44100 Hz, 0.4-second single-shot kick drums**.

---

## Architecture

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample rate | 44100 Hz | CD quality, industry standard for sample packs |
| `N_BAND` | 16 | Standard PQMF frequency bands |
| `RATIOS` | `[4, 4, 2]` | Compression ×512 → ~11.6 ms/frame, ~34 frames/kick. Default `[4,4,4,2]` would give only ~8 frames — too coarse for transient detail |
| `LATENT_SIZE` | 8 | Kicks are spectrally simpler than speech; 8 dimensions is sufficient |
| `CAPACITY` | 48 | Medium capacity — trains faster on small datasets vs default 64 |
| `discriminator_factors` | `[1,1,1]` | 3 scales matching `len(RATIOS)=3` |

---

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU (RTX 3090 / 4090 recommended for training; 8 GB+ VRAM)
- CUDA 12.x and matching PyTorch build

### Install

```bash
cd hardstyle-kick-rave
pip install -r requirements.txt
```

Verify the RAVE CLI is accessible:

```bash
rave --help
```

---

## Workflow

### 1. Add kick samples

Drop your hardstyle kick WAV/MP3/FLAC files into `data/raw/`.

**Recommendations:**
- 100 kicks minimum; 200–500 is ideal
- Consistent quality matters more than quantity — avoid clipping, silence-heavy files, or mixed sub-genres
- Full-length kicks work fine; the preprocessor aligns and trims them automatically

### 2. Preprocess

```bash
python scripts/preprocess.py
```

This does two things:

**Step A** — for each file in `data/raw/`:
- Resamples to 44100 Hz mono
- Detects the first transient onset and aligns to it (2 ms pre-roll)
- Trims or zero-pads to exactly 17640 samples (0.4 s)
- Peak-normalizes to −1 dBFS
- Saves 24-bit WAV to `data/processed/`

**Step B** — runs `rave preprocess` to build the LMDB training database at `data/kicks.mdb`.

Custom paths:

```bash
python scripts/preprocess.py --input my_kicks/ --processed data/processed --db data/kicks.mdb
```

### 3. Train

```bash
python scripts/train.py
```

With custom options:

```bash
python scripts/train.py --name kick_v1 --steps 800000 --gpu 0
```

| Flag | Default | Notes |
|------|---------|-------|
| `--steps` | 600000 | 400k–800k is good for 100–300 sample datasets |
| `--val-every` | 10000 | How often RAVE logs reconstructed audio |
| `--name` | kick_rave | Subdirectory name under `outputs/` |
| `--gpu` | 0 | GPU index |

Monitor training:

```bash
tensorboard --logdir outputs/
```

Training generates reconstructed audio samples every `--val-every` steps — listen to these to gauge quality.

**Estimated training time** on RTX 4090:
- 400k steps: ~20–30 hours
- 600k steps: ~35–45 hours

RAVE has two internal phases it transitions between automatically:
1. **Representation learning** — encoder + decoder learn the audio structure
2. **Adversarial fine-tuning** — GAN discriminators sharpen audio quality

### 4. Export

```bash
python scripts/export.py --run outputs/kick_rave
```

Produces `outputs/kick_rave/kick_rave.ts` — a self-contained TorchScript model for inference.

For real-time use (e.g. Max/MSP via nn~ or VST via RAVE2VST):

```bash
python scripts/export.py --run outputs/kick_rave --streaming
```

### 5. Generate

```bash
python scripts/generate.py --model outputs/kick_rave/kick_rave.ts --num 20
```

| Flag | Default | Notes |
|------|---------|-------|
| `--num` | 10 | Number of kicks to generate |
| `--temperature` | 1.0 | 0.5–0.8: conservative; 1.0: normal; 1.2–1.5: experimental |
| `--seed` | None | Fix for reproducible batches |
| `--output` | outputs/generated | Output directory |

Generated kicks are 24-bit WAV, peak-normalized to −1 dBFS, 0.4 s at 44100 Hz.

### 6. Interpolate between two kicks

Blend two kicks smoothly through latent space:

```bash
python scripts/interpolate.py \
    --model outputs/kick_rave/kick_rave.ts \
    --kick-a data/raw/kick_hard.wav \
    --kick-b data/raw/kick_distorted.wav \
    --steps 8
```

Outputs 8 WAV files stepping from `kick_hard` (α=0.0) to `kick_distorted` (α=1.0).

---

## Tips for Hardstyle Kicks

**Temperature tuning:**
- `0.7–0.9` — tight kicks, close to training distribution, low artifact risk
- `1.0` — balanced; default starting point
- `1.1–1.4` — more variation, may introduce distortion or unusual tones (often interesting)
- `>1.5` — unpredictable; mostly noise

**Common issues:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Muddy/washed-out low end | Temperature too high | Lower to 0.8–0.9 |
| Clicks at sample start | Onset detection misaligned | Inspect with `librosa.display`, or pass `--skip-rave-preprocess` and manually trim |
| Training loss stalls early | Dataset too small / all very similar kicks | Add more variety, or reduce `CAPACITY` in gin config to 32 |
| Generated kicks sound identical | Mode collapse — dataset too small for GAN phase | Add data augmentation (pitch shift ±1 semitone, slight time-stretch) |
| RAVE gin config errors on load | Parameter name changed between RAVE versions | Run `rave train --config configs/kick_rave.gin --print_gin` to see accepted params |

**Data augmentation** — if you have fewer than 100 kicks, augment in `scripts/preprocess.py` by adding pitch-shifted variants (±1 semitone with `librosa.effects.pitch_shift`) before the RAVE preprocess step.

---

## Project structure

```
hardstyle-kick-rave/
├── data/
│   ├── raw/               ← drop your kick samples here
│   ├── processed/         ← created by preprocess.py
│   └── kicks.mdb/         ← LMDB training database
├── configs/
│   └── kick_rave.gin      ← RAVE architecture config
├── scripts/
│   ├── preprocess.py      ← normalize / align / pad → LMDB
│   ├── train.py           ← launch RAVE training
│   ├── export.py          ← export checkpoint to .ts
│   ├── generate.py        ← sample new kicks from latent prior
│   └── interpolate.py     ← blend two kicks in latent space
└── outputs/
    └── kick_rave/         ← checkpoints + kick_rave.ts
```

---

## References

- [RAVE: A variational autoencoder for fast and high-quality neural audio synthesis](https://arxiv.org/abs/2111.05011) — Caillon & Esling, 2021
- [acids-ircam/RAVE](https://github.com/acids-ircam/RAVE) — official PyTorch implementation
- [nn~](https://github.com/acids-ircam/nn_tilde) — Max/MSP integration for real-time RAVE inference
