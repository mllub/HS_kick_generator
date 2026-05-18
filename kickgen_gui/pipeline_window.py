"""PipelineWindow — the main application window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import copy

import soundfile as sf  # type: ignore[import]

from kickgen.channel import Channel
from kickgen.pipeline import Pipeline
from kickgen_gui.audio import RenderWorker, play_audio
from kickgen_gui.channel_window import ChannelWindow
from kickgen_gui.serialization import load_pipeline, save_pipeline


# ---------------------------------------------------------------------------
# ChannelCard
# ---------------------------------------------------------------------------


class ChannelCard(QFrame):
    """A compact card widget representing a single channel in the pipeline view.

    Clicking the card opens the :class:`ChannelWindow` for that channel.
    """

    def __init__(
        self,
        channel_name: str,
        channel: Channel,
        pipeline_window: "PipelineWindow",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.channel_name = channel_name
        self.channel = channel
        self.pipeline_window = pipeline_window

        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(2)
        self.setMinimumWidth(160)
        self.setMaximumWidth(220)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setStyleSheet(
            "ChannelCard { background-color: #2b2b2b; border: 2px solid #555; "
            "border-radius: 6px; }"
            "ChannelCard:hover { border-color: #88aaff; }"
            "QLabel { color: #eeeeee; }"
        )
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        self._name_label = QLabel(channel_name)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._name_label)

        self._pan_label = QLabel()
        self._gain_label = QLabel()
        self._blocks_label = QLabel()
        for lbl in (self._pan_label, self._gain_label, self._blocks_label):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            layout.addWidget(lbl)

        layout.addStretch()

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self._mute_btn = QPushButton("Mute")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setChecked(channel.muted)
        self._mute_btn.setStyleSheet(
            "QPushButton { border: 1px solid #777; border-radius: 3px; padding: 2px 6px; }"
            "QPushButton:checked { background-color: #cc4444; color: white; border-color: #ff6666; }"
        )
        self._mute_btn.clicked.connect(self._on_mute_clicked)
        btn_layout.addWidget(self._mute_btn)

        dupe_btn = QPushButton("Duplicate")
        dupe_btn.setStyleSheet(
            "QPushButton { border: 1px solid #777; border-radius: 3px; padding: 2px 6px; }"
        )
        dupe_btn.clicked.connect(self._on_duplicate_clicked)
        btn_layout.addWidget(dupe_btn)

        layout.addWidget(btn_row)

        self._refresh()

    def _refresh(self) -> None:
        self._name_label.setText(self.channel_name)
        self._pan_label.setText(f"Pan: {self.channel.pan:+.2f}")
        self._gain_label.setText(f"Gain: {self.channel.gain_db:+.1f} dB")
        n = len(self.channel.blocks)
        self._blocks_label.setText(f"{n} block{'s' if n != 1 else ''}")
        self._mute_btn.setChecked(self.channel.muted)

    def _on_mute_clicked(self, checked: bool) -> None:
        self.channel.muted = checked
        self.pipeline_window._set_status(
            f"{self.channel_name}: {'muted' if checked else 'unmuted'}"
        )

    def _on_duplicate_clicked(self) -> None:
        self.pipeline_window.duplicate_channel(self.channel_name)

    def refresh(self) -> None:
        """Public refresh — called by PipelineWindow when the channel changes."""
        self._refresh()

    def mousePressEvent(self, event) -> None:
        self.pipeline_window.open_channel_window(self.channel_name)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# PipelineWindow
# ---------------------------------------------------------------------------


class PipelineWindow(QMainWindow):
    """Main application window — manages the whole Pipeline."""

    def __init__(self, pipeline: Pipeline, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self._open_channel_windows: dict[str, ChannelWindow] = {}
        self._worker: RenderWorker | None = None
        self._export_worker: RenderWorker | None = None
        self._export_path: str = ""

        self.setWindowTitle("KickGen")
        self.resize(900, 480)

        self._build_toolbar()
        self._build_central_area()
        self._build_status_bar()
        self._install_shortcut()

    # ---------------------------------------------------------------------- #
    # UI construction
    # ---------------------------------------------------------------------- #

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._play_btn = QPushButton("▶  Generate && Play")
        self._play_btn.clicked.connect(self.generate_and_play)
        toolbar.addWidget(self._play_btn)

        toolbar.addSeparator()

        self._export_btn = QPushButton("Export WAV")
        self._export_btn.clicked.connect(self.export_wav)
        toolbar.addWidget(self._export_btn)

        toolbar.addSeparator()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_pipeline)
        toolbar.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_pipeline)
        toolbar.addWidget(load_btn)

        toolbar.addSeparator()

        add_ch_btn = QPushButton("Add Channel")
        add_ch_btn.clicked.connect(self.add_channel)
        toolbar.addWidget(add_ch_btn)

        remove_ch_btn = QPushButton("Remove Channel")
        remove_ch_btn.clicked.connect(self.remove_channel)
        toolbar.addWidget(remove_ch_btn)

        toolbar.addSeparator()

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("padding-left: 8px; color: #aaaaaa;")
        toolbar.addWidget(self._status_label)

    def _build_central_area(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._cards_container = QWidget()
        self._cards_layout = QHBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(12, 12, 12, 12)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch()

        scroll.setWidget(self._cards_container)
        self.setCentralWidget(scroll)

        self._cards: dict[str, ChannelCard] = {}
        self._rebuild_cards()

    def _build_status_bar(self) -> None:
        self.setStatusBar(QStatusBar())

    def _install_shortcut(self) -> None:
        shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        shortcut.activated.connect(self.generate_and_play)

    # ---------------------------------------------------------------------- #
    # Card management
    # ---------------------------------------------------------------------- #

    def _rebuild_cards(self) -> None:
        """Re-build all channel cards from the current pipeline state."""
        # Remove existing cards from layout (keep stretch at the end)
        for card in self._cards.values():
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        for ch_name, channel in self.pipeline.channels:
            card = ChannelCard(ch_name, channel, self)
            # Insert before the trailing stretch
            stretch_idx = self._cards_layout.count() - 1
            self._cards_layout.insertWidget(stretch_idx, card)
            self._cards[ch_name] = card

    def _refresh_card(self, channel_name: str) -> None:
        if channel_name in self._cards:
            self._cards[channel_name].refresh()

    # ---------------------------------------------------------------------- #
    # Channel window management
    # ---------------------------------------------------------------------- #

    def open_channel_window(self, channel_name: str) -> None:
        """Open (or raise) the ChannelWindow for *channel_name*."""
        if channel_name in self._open_channel_windows:
            win = self._open_channel_windows[channel_name]
            win.raise_()
            win.activateWindow()
            return

        channel = self._find_channel(channel_name)
        if channel is None:
            return

        win = ChannelWindow(channel, channel_name, parent=self)
        win.channel_changed.connect(lambda: self._on_channel_changed(channel_name))
        win.setAttribute(Qt.WA_DeleteOnClose)
        win.destroyed.connect(lambda: self._open_channel_windows.pop(channel_name, None))
        win.show()
        self._open_channel_windows[channel_name] = win

    def _on_channel_changed(self, channel_name: str) -> None:
        self._refresh_card(channel_name)

    # ---------------------------------------------------------------------- #
    # Generate & play
    # ---------------------------------------------------------------------- #

    def generate_and_play(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._play_btn.setEnabled(False)
        self._set_status("Rendering…")

        self._worker = RenderWorker(self.pipeline, length_seconds=2.0, sr=44100)
        self._worker.finished.connect(self._on_render_done)
        self._worker.error.connect(self._on_render_error)
        self._worker.start()

    def _on_render_done(self, audio) -> None:
        try:
            play_audio(audio, sr=44100)
            self._set_status("Playing")
        except Exception as exc:
            self._set_status(f"Playback error: {exc}")
        self._play_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

    def _on_render_error(self, msg: str) -> None:
        self._set_status(f"Error: {msg}")
        self._play_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

    # ---------------------------------------------------------------------- #
    # Export WAV
    # ---------------------------------------------------------------------- #

    def export_wav(self) -> None:
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export WAV",
            "kick.wav",
            "WAV files (*.wav);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"
        self._export_path = path
        self._export_btn.setEnabled(False)
        self._play_btn.setEnabled(False)
        self._set_status("Rendering for export…")

        self._export_worker = RenderWorker(self.pipeline, length_seconds=2.0, sr=44100)
        self._export_worker.finished.connect(self._on_export_done)
        self._export_worker.error.connect(self._on_render_error)
        self._export_worker.start()

    def _on_export_done(self, audio) -> None:
        try:
            sf.write(self._export_path, audio, 44100, subtype="PCM_24")
            self._set_status(f"Exported: {self._export_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            self._set_status("Export failed")
        self._export_btn.setEnabled(True)
        self._play_btn.setEnabled(True)

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        self.statusBar().showMessage(text, 5000)

    # ---------------------------------------------------------------------- #
    # Add / Remove channels
    # ---------------------------------------------------------------------- #

    def add_channel(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Add Channel", "Channel name:", text="new_channel"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._find_channel(name) is not None:
            QMessageBox.warning(self, "Duplicate", f"Channel '{name}' already exists.")
            return
        new_channel = Channel([], pan=0.0, gain_db=0.0)
        self.pipeline.channels.append((name, new_channel))
        self._rebuild_cards()

    def duplicate_channel(self, source_name: str) -> None:
        source = self._find_channel(source_name)
        if source is None:
            return

        default_name = f"{source_name}_copy"
        # Auto-increment if the default name is already taken
        existing = {n for n, _ in self.pipeline.channels}
        if default_name in existing:
            i = 2
            while f"{source_name}_copy{i}" in existing:
                i += 1
            default_name = f"{source_name}_copy{i}"

        name, ok = QInputDialog.getText(
            self, "Duplicate Channel", "New channel name:", text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._find_channel(name) is not None:
            QMessageBox.warning(self, "Duplicate", f"Channel '{name}' already exists.")
            return

        new_channel = copy.deepcopy(source)

        # Insert immediately after the source channel
        idx = next(i for i, (n, _) in enumerate(self.pipeline.channels) if n == source_name)
        self.pipeline.channels.insert(idx + 1, (name, new_channel))
        self._rebuild_cards()
        self._set_status(f"Duplicated '{source_name}' → '{name}'")

    def remove_channel(self) -> None:
        if not self.pipeline.channels:
            return
        # Ask user which channel to remove via a simple dialog
        ch_names = [n for n, _ in self.pipeline.channels]
        name, ok = QInputDialog.getItem(
            self, "Remove Channel", "Select channel to remove:", ch_names, 0, False
        )
        if not ok or not name:
            return
        # Close the window if open
        if name in self._open_channel_windows:
            self._open_channel_windows[name].close()
        self.pipeline.channels = [
            (n, ch) for n, ch in self.pipeline.channels if n != name
        ]
        self._rebuild_cards()

    # ---------------------------------------------------------------------- #
    # Save / Load
    # ---------------------------------------------------------------------- #

    def save_pipeline(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline",
            "",
            "Kick Pipelines (*.kick.json);;All Files (*)",
        )
        if not path:
            return
        if not path.endswith(".kick.json"):
            path += ".kick.json"
        try:
            save_pipeline(self.pipeline, path)
            self._set_status(f"Saved: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def load_pipeline(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline",
            "",
            "Kick Pipelines (*.kick.json);;All Files (*)",
        )
        if not path:
            return
        try:
            new_pipeline = load_pipeline(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        # Close all open channel windows
        for win in list(self._open_channel_windows.values()):
            win.close()
        self._open_channel_windows.clear()

        # Replace pipeline in place so the window keeps the same reference
        self.pipeline.channels = new_pipeline.channels
        self.pipeline.master_gain_db = new_pipeline.master_gain_db
        self.pipeline.use_limiter = new_pipeline.use_limiter

        self._rebuild_cards()
        self._set_status(f"Loaded: {path}")

    # ---------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------- #

    def _find_channel(self, name: str) -> Channel | None:
        for n, ch in self.pipeline.channels:
            if n == name:
                return ch
        return None
