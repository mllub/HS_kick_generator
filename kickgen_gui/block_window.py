"""BlockWindow — auto-generated parameter editor for a single DSP block."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSlider,
    QWidget,
    QHBoxLayout,
)


class BlockWindow(QWidget):
    """Floating editor window for a single DSP block.

    Automatically builds a row of controls for every parameter exposed by
    ``block.get_params()`` / ``block.param_bounds()``.

    Parameters
    ----------
    block:
        The DSP block to edit.
    block_name:
        Logical name of the block (used for the window title).
    parent:
        Optional Qt parent widget.
    """

    params_changed = Signal()

    def __init__(self, block, block_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.block = block
        self.block_name = block_name

        block_type = type(block).__name__
        self.setWindowTitle(f"{block_type} — {block_name}")
        self.setMinimumWidth(400)

        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignRight)

        self._widgets: dict[str, QWidget] = {}
        self._guard = False  # re-entrancy guard for slider ↔ spinbox sync

        params = block.get_params()
        bounds = block.param_bounds()

        for param_name, current_value in params.items():
            bound = bounds.get(param_name)

            if isinstance(bound, list):
                # --- Discrete parameter: show a QComboBox ---
                combo = QComboBox()
                for option in bound:
                    combo.addItem(str(option))
                # current_value is stored as a float index
                index = int(round(float(current_value)))
                index = max(0, min(index, len(bound) - 1))
                combo.setCurrentIndex(index)

                # Capture param_name in closure
                def make_combo_handler(pname, cb):
                    def on_combo_changed(idx):
                        self.block.set_params(**{pname: float(idx)})
                        self.params_changed.emit()
                    cb.currentIndexChanged.connect(on_combo_changed)

                make_combo_handler(param_name, combo)
                self._widgets[param_name] = combo
                layout.addRow(QLabel(param_name), combo)

            elif isinstance(bound, tuple) and len(bound) == 2:
                # --- Continuous parameter: slider + spinbox ---
                min_val, max_val = float(bound[0]), float(bound[1])
                step = (max_val - min_val) / 1000.0

                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)

                slider = QSlider(Qt.Horizontal)
                slider.setRange(0, 1000)
                slider.setMinimumWidth(160)

                spinbox = QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setSingleStep(step)
                spinbox.setDecimals(4)
                spinbox.setMinimumWidth(90)

                # Set initial value
                fval = float(current_value)
                fval = max(min_val, min(max_val, fval))
                slider_pos = int(round((fval - min_val) / (max_val - min_val) * 1000))
                slider.setValue(slider_pos)
                spinbox.setValue(fval)

                row_layout.addWidget(slider)
                row_layout.addWidget(spinbox)

                # Wire up sync + parameter update
                def make_continuous_handlers(pname, sl, sb, lo, hi):
                    def on_slider_changed(pos):
                        if self._guard:
                            return
                        self._guard = True
                        val = lo + (pos / 1000.0) * (hi - lo)
                        sb.setValue(val)
                        self.block.set_params(**{pname: val})
                        self.params_changed.emit()
                        self._guard = False

                    def on_spinbox_changed(val):
                        if self._guard:
                            return
                        self._guard = True
                        pos = int(round((val - lo) / (hi - lo) * 1000))
                        pos = max(0, min(1000, pos))
                        sl.setValue(pos)
                        self.block.set_params(**{pname: val})
                        self.params_changed.emit()
                        self._guard = False

                    sl.valueChanged.connect(on_slider_changed)
                    sb.valueChanged.connect(on_spinbox_changed)

                make_continuous_handlers(param_name, slider, spinbox, min_val, max_val)
                self._widgets[param_name] = row_widget
                layout.addRow(QLabel(param_name), row_widget)

            else:
                # Fallback: read-only label
                lbl = QLabel(str(current_value))
                self._widgets[param_name] = lbl
                layout.addRow(QLabel(param_name), lbl)

        self.setLayout(layout)

    def refresh(self) -> None:
        """Refresh all displayed values from the block's current state."""
        params = self.block.get_params()
        bounds = self.block.param_bounds()
        self._guard = True
        for param_name, current_value in params.items():
            widget = self._widgets.get(param_name)
            if widget is None:
                continue
            bound = bounds.get(param_name)
            if isinstance(bound, list) and isinstance(widget, QComboBox):
                index = int(round(float(current_value)))
                index = max(0, min(index, len(bound) - 1))
                widget.setCurrentIndex(index)
            elif isinstance(bound, tuple):
                min_val, max_val = float(bound[0]), float(bound[1])
                # widget is a QWidget container; find child slider and spinbox
                sl = widget.findChild(QSlider)
                sb = widget.findChild(QDoubleSpinBox)
                if sl is not None and sb is not None:
                    fval = max(min_val, min(max_val, float(current_value)))
                    pos = int(round((fval - min_val) / (max_val - min_val) * 1000))
                    sl.setValue(pos)
                    sb.setValue(fval)
        self._guard = False
