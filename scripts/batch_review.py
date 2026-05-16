"""
Visually review kick samples in batches of 10.

Shows 10 waveform subplots. Press a number to delete that sample and replace
it with the next one from the queue. Press SPACE when all 10 are OK.

Controls:
    1-9, 0  — delete that subplot (0 = subplot 10), replace with next sample
    SPACE   — all 10 OK, advance to next batch
    Q       — quit

Usage:
    python scripts/batch_review.py
    python scripts/batch_review.py --input data/raw
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import librosa
except ImportError:
    print("Error: librosa not installed.")
    sys.exit(1)

TARGET_SR = 44100
BATCH_SIZE = 10
EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".ogg"}


class BatchReviewer:
    def __init__(self, files: list[Path]):
        self.queue = list(files)
        self.batch: list[Path | None] = []
        self.history: list[list[Path | None]] = []
        self.cache: dict[Path, np.ndarray] = {}
        self.deleted = 0
        self.total = len(files)

        self.fig, self.axes = plt.subplots(BATCH_SIZE, 1, figsize=(14, 18))
        self.prog_ax = self.fig.add_axes([0.05, 0.01, 0.90, 0.012])
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._fill_batch()
        self._preload_batch()
        self._draw()
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.show()

    def _load(self, path: Path) -> np.ndarray:
        if path not in self.cache:
            try:
                audio, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
            except Exception:
                audio = np.zeros(TARGET_SR // 10, dtype=np.float32)
            self.cache[path] = audio
        return self.cache[path]

    def _preload_batch(self) -> None:
        for path in self.batch:
            if path is not None:
                self._load(path)

    def _fill_batch(self) -> None:
        self.batch = []
        for _ in range(BATCH_SIZE):
            self.batch.append(self.queue.pop(0) if self.queue else None)

    def _draw(self) -> None:
        reviewed = self.total - len(self.queue) - sum(p is not None for p in self.batch)
        self.fig.suptitle(
            f"Batch review  —  {reviewed}/{self.total} reviewed  |  {self.deleted} deleted\n"
            "1–9 / 0 = delete subplot   SPACE = next batch   ← = previous batch   Q = quit",
            fontsize=9,
        )

        for i, ax in enumerate(self.axes):
            ax.cla()
            path = self.batch[i]
            label = str(i + 1) if i < 9 else "0"

            if path is not None:
                audio = self._load(path)
                t = np.arange(len(audio)) / TARGET_SR
                ax.plot(t, audio, linewidth=0.4, color="steelblue")
                ax.set_title(f"{label}  —  {path.name}", fontsize=7, loc="left", pad=2)
                ax.set_xlim(0, 0.6)
            else:
                ax.set_title(f"{label}  —  (end of files)", fontsize=7, loc="left", pad=2)

            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(labelsize=6)

        self._draw_progress(reviewed)
        self.fig.canvas.draw_idle()

    def _draw_progress(self, reviewed: int) -> None:
        self.prog_ax.cla()
        progress = reviewed / self.total if self.total > 0 else 0
        self.prog_ax.barh(0, 1, height=1, color="#e0e0e0")
        self.prog_ax.barh(0, progress, height=1, color="steelblue")
        self.prog_ax.set_xlim(0, 1)
        self.prog_ax.set_ylim(-0.5, 0.5)
        self.prog_ax.set_xticks([])
        self.prog_ax.set_yticks([])
        self.prog_ax.text(0.5, 0, f"{reviewed} / {self.total}  ({progress * 100:.1f}%)",
                          ha="center", va="center", fontsize=8, color="white" if progress > 0.5 else "black")

    def _on_key(self, event) -> None:
        key = event.key

        if key == " ":
            if all(p is None for p in self.batch):
                print("All done.")
                plt.close()
                return
            self.history.append(list(self.batch))
            self._fill_batch()
            if all(p is None for p in self.batch):
                print(f"All done. {self.deleted} deleted.")
                plt.close()
                return
            self._preload_batch()
            self._draw()

        elif key == "left":
            if not self.history:
                return
            # Return current non-None files to the front of the queue
            returning = [p for p in reversed(self.batch) if p is not None]
            self.queue = returning + self.queue
            self.batch = self.history.pop()
            self._preload_batch()
            self._draw()

        elif key in "1234567890":
            idx = (int(key) - 1) % BATCH_SIZE  # '0' maps to index 9
            path = self.batch[idx]
            if path is None:
                return
            try:
                path.unlink()
                self.deleted += 1
                print(f"Deleted [{idx + 1 if idx < 9 else 0}] {path.name}")
            except Exception as e:
                print(f"Could not delete {path.name}: {e}")
            if path in self.cache:
                del self.cache[path]
            replacement = self.queue.pop(0) if self.queue else None
            self.batch[idx] = replacement
            if replacement is not None:
                self._load(replacement)
            self._draw()

        elif key.lower() == "q":
            print(f"Quit. {self.deleted} deleted.")
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Batch-review kick samples visually")
    parser.add_argument("--input", default="data/raw", help="Directory to scan")
    args = parser.parse_args()

    input_dir = Path(args.input)
    files = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in EXTENSIONS)

    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files. Opening batch review...")
    print("Controls: 1–9/0 = delete subplot   SPACE = all OK   Q = quit\n")
    BatchReviewer(files)


if __name__ == "__main__":
    main()
