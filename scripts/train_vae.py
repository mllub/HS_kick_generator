"""
Train a KickVAE on log-mel spectrograms of kick drum samples.

Loads data/processed/kicks.pt, computes mel spectrograms, trains the VAE and
shows a live reconstruction-loss plot that updates every 5 epochs.

Usage:
    python scripts/train_vae.py
    python scripts/train_vae.py --mode vae --latent-dim 32 --epochs 200
    python scripts/train_vae.py --mode ae
    python scripts/train_vae.py --mode vae_fixed --beta 0.5
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.kick_vae import KickVAE

TARGET_SR = 44100
N_MELS    = 128
N_FFT     = 1024
HOP       = 256
LOG_EPS   = 1e-9


# ── Mel spectrogram helpers ────────────────────────────────────────────────

def build_mel_transform(device: torch.device) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
    ).to(device)


def to_log_mel(waveforms: torch.Tensor, mel_tf: T.MelSpectrogram) -> torch.Tensor:
    """(N, 1, T) → (N, 1, n_mels, n_frames), log-scaled and normalised to [-1, 1]."""
    # waveforms: (N, 1, T) → squeeze channel for transform → (N, T)
    specs = mel_tf(waveforms.squeeze(1))          # (N, n_mels, n_frames)
    specs = torch.log(specs + LOG_EPS)            # log amplitude
    specs = specs.unsqueeze(1)                    # (N, 1, n_mels, n_frames)
    # Normalise globally to [-1, 1]
    s_min = specs.min()
    s_max = specs.max()
    specs = 2 * (specs - s_min) / (s_max - s_min + 1e-9) - 1
    return specs


# ── Keyboard stop helper ──────────────────────────────────────────────────

def _kbhit() -> bool:
    """Return True if a key has been pressed (non-blocking)."""
    if os.name == "nt":
        import msvcrt
        return msvcrt.kbhit()  # type: ignore[attr-defined]
    import select
    return bool(select.select([sys.stdin], [], [], 0)[0])


def _getch() -> str:
    """Read one character from stdin (assumes a key is available)."""
    if os.name == "nt":
        import msvcrt
        return msvcrt.getch().decode("utf-8", errors="ignore")  # type: ignore[attr-defined]
    return sys.stdin.read(1)


@contextmanager
def _cbreak_stdin():
    """Put the terminal in cbreak mode so single keypresses are readable.

    Falls back silently if stdin is not a tty (e.g. piped input or Windows).
    """
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return
    import termios, tty
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


# ── Live plot ──────────────────────────────────────────────────────────────

class LivePlot:
    def __init__(self, mode: str, out_dir: Path):
        self.out_dir = out_dir
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        (self.ax_loss, self.ax_orig,     self.ax_recon), \
        (self.ax_kl,   self.ax_val_orig, self.ax_val_recon) = axes

        self.ax_loss.set_ylabel("Recon loss (L1)")
        self.ax_loss.set_title(f"Loss  —  mode: {mode}")
        self.train_line, = self.ax_loss.plot([], [], label="Train", color="steelblue")
        self.val_line,   = self.ax_loss.plot([], [], label="Val",   color="tomato")
        self.ax_loss.legend()
        self.ax_loss.set_xticklabels([])
        self.ax_loss.grid(True, linestyle="--", alpha=0.4)

        self.ax_kl.set_xlabel("Epoch")
        self.ax_kl.set_ylabel("KL loss")
        self.train_kl_line, = self.ax_kl.plot([], [], label="Train", color="steelblue")
        self.val_kl_line,   = self.ax_kl.plot([], [], label="Val",   color="tomato")
        self.ax_kl.legend()
        self.ax_kl.grid(True, linestyle="--", alpha=0.4)

        self.fig.tight_layout()

        self.epochs: list[int]      = []
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.train_kls:    list[float] = []
        self.val_kls:      list[float] = []

    def record(self, epoch: int, train_recon: float, val_recon: float,
               train_kl: float, val_kl: float) -> None:
        self.epochs.append(epoch)
        self.train_losses.append(train_recon)
        self.val_losses.append(val_recon)
        self.train_kls.append(train_kl)
        self.val_kls.append(val_kl)

    def redraw(self, model: "KickVAE", train_sample: torch.Tensor,
               val_sample: torch.Tensor, device: torch.device) -> None:
        # --- loss plots ---
        self.train_line.set_data(self.epochs, self.train_losses)
        self.val_line.set_data(self.epochs, self.val_losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.train_kl_line.set_data(self.epochs, self.train_kls)
        self.val_kl_line.set_data(self.epochs, self.val_kls)
        self.ax_kl.relim()
        self.ax_kl.autoscale_view()

        # --- mel plots ---
        model.eval()
        with torch.no_grad():
            train_recon, _, _ = model(train_sample.unsqueeze(0).to(device))
            val_recon,   _, _ = model(val_sample.unsqueeze(0).to(device))

        for (ax_orig, ax_recon, orig, recon, row_label) in [
            (self.ax_orig,     self.ax_recon,     train_sample, train_recon, "Train"),
            (self.ax_val_orig, self.ax_val_recon, val_sample,   val_recon,   "Val"),
        ]:
            orig_np  = orig.squeeze().cpu().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            vmin, vmax = orig_np.min(), orig_np.max()
            for ax, img, title in [
                (ax_orig,  orig_np,  f"{row_label} — original"),
                (ax_recon, recon_np, f"{row_label} — reconstruction"),
            ]:
                ax.cla()
                ax.imshow(img, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma")
                ax.set_title(title)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Mel bin")

        model.train()
        self.fig.tight_layout()
        epoch = self.epochs[-1] if self.epochs else 0
        plot_path = self.out_dir / f"plot_ep{epoch:04d}.png"
        self.fig.savefig(plot_path, dpi=100)
        print(f"  Plot saved → {plot_path}")


# ── Training loop ──────────────────────────────────────────────────────────

def run_epoch(model: KickVAE, loader: DataLoader, optimizer, device: torch.device,
              beta: float, train: bool) -> tuple[float, float]:
    model.train(train)
    total_recon = 0.0
    total_kl    = 0.0
    with torch.set_grad_enabled(train):
        for (x,) in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            recon_loss = F.l1_loss(recon, x)
            kl_loss    = model.kl_loss(mu, logvar)
            loss       = recon_loss + beta * kl_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_recon += recon_loss.item()
            total_kl    += kl_loss.item()
    n = len(loader)
    return total_recon / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Train KickVAE")
    parser.add_argument("--data",       default="data/processed/kicks.pt")
    parser.add_argument("--out",        default="outputs/vae",     help="Checkpoint directory")
    parser.add_argument("--mode",       default="vae",             choices=["ae", "vae_fixed", "vae"])
    parser.add_argument("--latent-dim", type=int,   default=64,
                        help="Latent channels at bottleneck (fully-conv model)")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0,    help="AdamW weight decay")
    parser.add_argument("--beta",       type=float, default=1.0,   help="KL weight (beta-VAE)")
    parser.add_argument("--val-split",  type=float, default=0.1,   help="Fraction of data for validation")
    parser.add_argument("--plot-every",       type=int, default=5,  help="Save plot every N epochs")
    parser.add_argument("--checkpoint-every", type=int, default=5,  help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cpu":
        n_threads = os.cpu_count() or 1
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        print(f"CPU threads : {n_threads}")
    print(f"Mode   : {args.mode}")

    # ── Load data ──────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}  (run scripts/preprocess.py first)")
        sys.exit(1)

    print(f"Loading {data_path} ...")
    waveforms = torch.load(data_path, weights_only=True)   # (N, 1, T)
    print(f"  Waveforms : {tuple(waveforms.shape)}")

    # ── Mel spectrograms ───────────────────────────────────────────────────
    print("Computing log-mel spectrograms ...")
    mel_tf = build_mel_transform(device)
    waveforms = waveforms.to(device)
    with torch.no_grad():
        specs = to_log_mel(waveforms, mel_tf)              # (N, 1, n_mels, n_frames)
    print(f"  Spectrograms : {tuple(specs.shape)}")
    n_frames = specs.shape[-1]

    # ── Dataset split ──────────────────────────────────────────────────────
    dataset  = TensorDataset(specs.cpu())
    n_val    = max(1, int(len(dataset) * args.val_split))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"  Train: {n_train}  |  Val: {n_val}")

    # ── Resume from checkpoint? ────────────────────────────────────────────
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    ckpt        = None

    answer = input("\nLoad latest checkpoint? [y/n]: ").strip().lower()
    if answer == "y":
        candidates = sorted(out_dir.glob("kick_vae_ep*.pt"))
        if candidates:
            ckpt_path = candidates[-1]
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_mode = ckpt.get("args", {}).get("mode", "unknown") if isinstance(ckpt, dict) else "unknown"
            print(f"  Loaded {ckpt_path.name}  (saved mode: {saved_mode})")
        else:
            print(f"  No checkpoints found in {out_dir} — starting fresh.")

    # ── Mode selection ─────────────────────────────────────────────────────
    valid_modes = ("ae", "vae_fixed", "vae")
    default_mode = (
        ckpt.get("args", {}).get("mode", args.mode)
        if isinstance(ckpt, dict) else args.mode
    )
    while True:
        raw = input(f"Mode [ae / vae_fixed / vae]  (default: {default_mode}): ").strip().lower()
        if raw == "":
            chosen_mode = default_mode
            break
        if raw in valid_modes:
            chosen_mode = raw
            break
        print(f"  Invalid mode. Choose from: {', '.join(valid_modes)}")
    print(f"  Using mode: {chosen_mode}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = KickVAE(
        mode=chosen_mode,
        latent_dim=args.latent_dim,
        n_mels=N_MELS,
        n_frames=n_frames,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if ckpt is not None:
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"  Resumed from epoch {ckpt['epoch']}")
        else:
            model.load_state_dict(ckpt)
            print("  Loaded weights from final checkpoint")

    # ── Training ───────────────────────────────────────────────────────────
    # Fixed random samples used for the mel reconstruction subplots
    train_rng_idx = torch.randint(len(train_ds), (1,)).item()
    val_rng_idx   = torch.randint(len(val_ds),   (1,)).item()
    fixed_sample     = train_ds[train_rng_idx][0]    # (1, n_mels, n_frames)
    fixed_val_sample = val_ds[val_rng_idx][0]        # (1, n_mels, n_frames)

    plot = LivePlot(chosen_mode, out_dir)
    print(f"\n{'Epoch':>6}  {'Train recon':>12}  {'Train KL':>10}  {'Val recon':>10}  {'Val KL':>8}")
    print("-" * 58)
    print("  Press SPACE to stop early and save  |  ENTER to save checkpoint now and keep training.\n")

    plot.redraw(model, fixed_sample, fixed_val_sample, device)

    with _cbreak_stdin():
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_recon, train_kl = run_epoch(model, train_loader, optimizer, device, args.beta, train=True)
            val_recon,   val_kl   = run_epoch(model, val_loader,   optimizer, device, args.beta, train=False)

            print(f"{epoch:>6}  {train_recon:>12.6f}  {train_kl:>10.6f}  {val_recon:>10.6f}  {val_kl:>8.6f}")

            plot.record(epoch, train_recon, val_recon, train_kl, val_kl)
            if epoch % args.plot_every == 0:
                plot.redraw(model, fixed_sample, fixed_val_sample, device)
            if epoch % args.checkpoint_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                }, out_dir / f"kick_vae_ep{epoch:04d}.pt")

            if _kbhit():
                ch = _getch()
                if ch == " ":
                    print("\n  Spacebar pressed — stopping early.")
                    break
                elif ch in ("\r", "\n"):
                    print(f"\n  Enter pressed — saving checkpoint at epoch {epoch}.")
                    plot.redraw(model, fixed_sample, fixed_val_sample, device)
                    torch.save({
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "args": vars(args),
                    }, out_dir / f"kick_vae_ep{epoch:04d}.pt")

    torch.save(model.state_dict(), out_dir / "kick_vae_final.pt")
    print(f"\nSaved final model → {out_dir / 'kick_vae_final.pt'}")
    plot.fig.savefig(out_dir / "plot_final.png", dpi=100)
    print(f"Final plot saved → {out_dir / 'plot_final.png'}")
    plt.close(plot.fig)


if __name__ == "__main__":
    main()
