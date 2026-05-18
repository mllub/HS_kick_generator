"""Entry point for the kickgen GUI."""

import sys

from PySide6.QtWidgets import QApplication

from kickgen.blocks import Compressor, KickSynth, ParametricEQ, Waveshaper
from kickgen.channel import Channel
from kickgen.pipeline import Pipeline
from kickgen_gui.pipeline_window import PipelineWindow


def make_default_pipeline() -> Pipeline:
    """Build a simple default pipeline for first launch."""
    tail = Channel(
        [
            ("kick", KickSynth()),
            ("eq", ParametricEQ(n_bands=2)),
            ("shaper", Waveshaper()),
            ("comp", Compressor()),
        ],
        pan=0.0,
        gain_db=0.0,
    )
    return Pipeline([("tail", tail)], master_gain_db=0.0)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("KickGen")
    pipeline = make_default_pipeline()
    window = PipelineWindow(pipeline)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
