"""Smoke test: GUI constructs without crashing."""

pytestqt = pytest = None
try:
    import pytest
    pytestqt = pytest.importorskip("pytestqt")
except ImportError:
    pass


def test_pipeline_window_constructs(qtbot):
    from kickgen_gui.main import make_default_pipeline
    from kickgen_gui.pipeline_window import PipelineWindow

    pipeline = make_default_pipeline()
    window = PipelineWindow(pipeline)
    qtbot.addWidget(window)
    window.show()
    # just check it doesn't crash
