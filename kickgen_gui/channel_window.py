"""ChannelWindow — editor for a single Channel object."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from kickgen.blocks import KickSynth
from kickgen.channel import Channel
from kickgen.registry import BLOCK_REGISTRY
from kickgen_gui.block_window import BlockWindow
from kickgen_gui.kicksynth_window import KickSynthWindow


class ChannelWindow(QWidget):
    """Floating editor window for a single :class:`~kickgen.channel.Channel`.

    Parameters
    ----------
    channel:
        The Channel object to edit.
    channel_name:
        Logical name used in the Pipeline's channel list (used for the title).
    parent:
        Optional Qt parent widget.
    """

    channel_changed = Signal()

    def __init__(
        self,
        channel: Channel,
        channel_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.Window)
        self.channel = channel
        self.channel_name = channel_name
        self._open_windows: list[QWidget] = []
        self._guard = False

        self.setWindowTitle(f"Channel — {channel_name}")
        self.setMinimumWidth(360)

        root_layout = QVBoxLayout(self)

        # ------------------------------------------------------------------ #
        # Top section: name, pan, gain
        # ------------------------------------------------------------------ #
        top_group = QGroupBox("Channel settings")
        form = QFormLayout(top_group)
        form.setLabelAlignment(Qt.AlignRight)

        # Name
        self._name_edit = QLineEdit(channel_name)
        self._name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("Name", self._name_edit)

        # Pan
        pan_widget, self._pan_slider, self._pan_spinbox = self._make_slider_spinbox(
            -1.0, 1.0, channel.pan, "pan"
        )
        form.addRow("Pan", pan_widget)

        # Gain dB
        gain_widget, self._gain_slider, self._gain_spinbox = self._make_slider_spinbox(
            -40.0, 12.0, channel.gain_db, "gain_db"
        )
        form.addRow("Gain (dB)", gain_widget)

        root_layout.addWidget(top_group)

        # ------------------------------------------------------------------ #
        # Block list
        # ------------------------------------------------------------------ #
        block_group = QGroupBox("Blocks")
        block_layout = QVBoxLayout(block_group)

        self._block_list = QListWidget()
        self._block_list.setMinimumHeight(120)
        self._block_list.itemDoubleClicked.connect(self._on_block_double_clicked)
        block_layout.addWidget(self._block_list)

        # Add-block row
        add_row = QHBoxLayout()
        self._add_combo = QComboBox()
        for name in BLOCK_REGISTRY:
            self._add_combo.addItem(name)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_block)
        add_row.addWidget(self._add_combo, stretch=1)
        add_row.addWidget(add_btn)
        block_layout.addLayout(add_row)

        # Action buttons row
        btn_row = QHBoxLayout()
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._on_remove_block)
        up_btn = QPushButton("Move Up")
        up_btn.clicked.connect(self._on_move_up)
        down_btn = QPushButton("Move Down")
        down_btn.clicked.connect(self._on_move_down)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(down_btn)
        block_layout.addLayout(btn_row)

        root_layout.addWidget(block_group)
        self.setLayout(root_layout)

        self._refresh_block_list()

    # ---------------------------------------------------------------------- #
    # Helper: slider + spinbox pair
    # ---------------------------------------------------------------------- #

    def _make_slider_spinbox(
        self,
        min_val: float,
        max_val: float,
        current: float,
        attr_name: str,
    ) -> tuple[QWidget, QSlider, QDoubleSpinBox]:
        """Return (container, slider, spinbox) wired together and to *channel*."""
        step = (max_val - min_val) / 1000.0

        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setMinimumWidth(120)

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setMinimumWidth(80)

        fval = max(min_val, min(max_val, float(current)))
        pos = int(round((fval - min_val) / (max_val - min_val) * 1000))
        slider.setValue(pos)
        spinbox.setValue(fval)

        row.addWidget(slider)
        row.addWidget(spinbox)

        def on_slider(pos):
            if self._guard:
                return
            self._guard = True
            val = min_val + (pos / 1000.0) * (max_val - min_val)
            spinbox.setValue(val)
            setattr(self.channel, attr_name, val)
            self.channel_changed.emit()
            self._guard = False

        def on_spinbox(val):
            if self._guard:
                return
            self._guard = True
            pos = int(round((val - min_val) / (max_val - min_val) * 1000))
            pos = max(0, min(1000, pos))
            slider.setValue(pos)
            setattr(self.channel, attr_name, val)
            self.channel_changed.emit()
            self._guard = False

        slider.valueChanged.connect(on_slider)
        spinbox.valueChanged.connect(on_spinbox)

        return container, slider, spinbox

    # ---------------------------------------------------------------------- #
    # Block list helpers
    # ---------------------------------------------------------------------- #

    def _refresh_block_list(self) -> None:
        self._block_list.clear()
        for blk_name, block in self.channel.blocks:
            type_name = type(block).__name__
            self._block_list.addItem(f"[{type_name}] {blk_name}")

    def _selected_index(self) -> int:
        row = self._block_list.currentRow()
        return row

    # ---------------------------------------------------------------------- #
    # Slots
    # ---------------------------------------------------------------------- #

    def _on_name_changed(self, text: str) -> None:
        # Only update the display name stored on the channel; the Pipeline-level
        # name is managed by PipelineWindow (it owns the (name, channel) tuple).
        self.setWindowTitle(f"Channel — {text}")
        self.channel_changed.emit()

    def _on_block_double_clicked(self, item) -> None:
        idx = self._block_list.row(item)
        if 0 <= idx < len(self.channel.blocks):
            blk_name, block = self.channel.blocks[idx]
            if isinstance(block, KickSynth):
                win = KickSynthWindow(block, blk_name, parent=self)
            else:
                win = BlockWindow(block, blk_name, parent=self)
            win.params_changed.connect(self.channel_changed)
            win.show()
            self._open_windows.append(win)

    def _on_add_block(self) -> None:
        selected_type = self._add_combo.currentText()
        block_cls = BLOCK_REGISTRY[selected_type]
        block = block_cls()
        idx = len(self.channel.blocks)
        auto_name = f"{selected_type.lower()}_{idx}"
        self.channel.blocks.append((auto_name, block))
        self._refresh_block_list()
        self.channel_changed.emit()

    def _on_remove_block(self) -> None:
        idx = self._selected_index()
        if 0 <= idx < len(self.channel.blocks):
            del self.channel.blocks[idx]
            self._refresh_block_list()
            self.channel_changed.emit()

    def _on_move_up(self) -> None:
        idx = self._selected_index()
        if idx > 0:
            lst = self.channel.blocks
            lst[idx - 1], lst[idx] = lst[idx], lst[idx - 1]
            self._refresh_block_list()
            self._block_list.setCurrentRow(idx - 1)
            self.channel_changed.emit()

    def _on_move_down(self) -> None:
        idx = self._selected_index()
        lst = self.channel.blocks
        if 0 <= idx < len(lst) - 1:
            lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
            self._refresh_block_list()
            self._block_list.setCurrentRow(idx + 1)
            self.channel_changed.emit()

    def get_current_name(self) -> str:
        """Return the name currently shown in the name field."""
        return self._name_edit.text()

    def refresh_from_channel(self) -> None:
        """Sync displayed values from the channel object (e.g. after external change)."""
        self._refresh_block_list()
        # Update sliders / spinboxes
        self._guard = True
        pan = self.channel.pan
        pan_pos = int(round((pan - (-1.0)) / 2.0 * 1000))
        self._pan_slider.setValue(max(0, min(1000, pan_pos)))
        self._pan_spinbox.setValue(pan)

        gain = self.channel.gain_db
        gain_pos = int(round((gain - (-40.0)) / 52.0 * 1000))
        self._gain_slider.setValue(max(0, min(1000, gain_pos)))
        self._gain_spinbox.setValue(gain)
        self._guard = False
