"""Worker thread for rendering and playing audio."""

from PySide6.QtCore import QThread, Signal
import numpy as np


class RenderWorker(QThread):
    """Renders the pipeline in a background thread."""

    finished = Signal(object)  # emits stereo float32 ndarray
    error = Signal(str)

    def __init__(self, pipeline, length_seconds: float = 2.0, sr: int = 44100):
        super().__init__()
        self.pipeline = pipeline
        self.length_seconds = length_seconds
        self.sr = sr

    def run(self):
        try:
            audio = self.pipeline.render(self.length_seconds, self.sr)
            self.finished.emit(audio)
        except Exception as e:
            self.error.emit(str(e))


def play_audio(audio: np.ndarray, sr: int = 44100) -> None:
    """Play a stereo float32 array via sounddevice (non-blocking)."""
    import sounddevice as sd  # type: ignore[import]
    sd.stop()
    sd.play(audio, samplerate=sr)
