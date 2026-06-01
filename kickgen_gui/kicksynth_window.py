"""Custom editor window for KickSynth blocks.

Left panel: global params (freq_scale dial, length slider), piecewise-linear
envelope editors (pitch + amplitude), harmonic controls.

Right panel: live matplotlib canvas with two dark subplots.  Orange knot dots
can be dragged directly on the plot.  The first knot is always pinned at t=0
and the last at t=length; only their value (y) can be changed.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDial,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── low-level helpers ─────────────────────────────────────────────────────────

def _make_spinbox(lo: float, hi: float, val: float,
                  decimals: int = 3, step: float | None = None) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step if step is not None else (hi - lo) / 100.0)
    sb.setValue(val)
    sb.setMinimumWidth(72)
    return sb


def _sl_to_val(pos: int, lo: float, hi: float, log: bool = False) -> float:
    t = pos / 1000.0
    if log and lo > 0:
        return 10.0 ** (math.log10(lo) + t * (math.log10(hi) - math.log10(lo)))
    return lo + t * (hi - lo)


def _val_to_sl(val: float, lo: float, hi: float, log: bool = False) -> int:
    if log and lo > 0:
        t = (math.log10(max(val, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    else:
        t = (val - lo) / (hi - lo)
    return int(round(max(0.0, min(1.0, t)) * 1000))


# ── Envelope knot-point row ───────────────────────────────────────────────────

class _EnvPointRow(QWidget):
    """Single (time, value) knot-point editor.

    When *fixed_t* is True the time spinbox is disabled — used for the first
    (t=0) and last (t=length) points.
    """

    changed = Signal()

    def __init__(
        self,
        pt_idx: int,
        t: float,
        v: float,
        t_range: tuple[float, float],
        v_range: tuple[float, float],
        t_label: str = "t (s)",
        v_label: str = "val",
        fixed_t: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._guard = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        row.addWidget(QLabel(f"#{pt_idx}"))

        row.addWidget(QLabel(t_label))
        self._t_sb = _make_spinbox(*t_range, t, decimals=3,
                                   step=(t_range[1] - t_range[0]) / 200)
        self._t_sb.setEnabled(not fixed_t)
        row.addWidget(self._t_sb)

        row.addWidget(QLabel(v_label))
        self._v_sb = _make_spinbox(*v_range, v, decimals=3,
                                   step=(v_range[1] - v_range[0]) / 200)
        row.addWidget(self._v_sb)

        self._t_sb.valueChanged.connect(self._emit)
        self._v_sb.valueChanged.connect(self._emit)

    def _emit(self) -> None:
        if not self._guard:
            self.changed.emit()

    def t(self) -> float:
        return self._t_sb.value()

    def v(self) -> float:
        return self._v_sb.value()

    def set_tv(self, t: float, v: float) -> None:
        """Update both fields without emitting changed."""
        self._guard = True
        self._t_sb.setValue(t)
        self._v_sb.setValue(v)
        self._guard = False

    def set_t(self, t: float) -> None:
        """Update only the time field without emitting changed."""
        self._guard = True
        self._t_sb.setValue(t)
        self._guard = False

    def set_v(self, v: float) -> None:
        """Update only the value field without emitting changed."""
        self._guard = True
        self._v_sb.setValue(v)
        self._guard = False


# ── Harmonic row ──────────────────────────────────────────────────────────────

class _HarmonicRow(QWidget):
    """Amplitude + phase controls for one harmonic."""

    changed = Signal()

    def __init__(self, harmonic_num: int, amp: float, phase_deg: float,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._guard = False

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        row.addWidget(QLabel(f"H{harmonic_num}"))

        row.addWidget(QLabel("amp"))
        self._amp_sl = QSlider(Qt.Horizontal)
        self._amp_sl.setRange(0, 1000)
        self._amp_sl.setValue(_val_to_sl(amp, 0.0, 1.0))
        self._amp_sl.setMinimumWidth(80)
        self._amp_sb = _make_spinbox(0.0, 1.0, amp, decimals=3, step=0.01)
        row.addWidget(self._amp_sl)
        row.addWidget(self._amp_sb)

        row.addWidget(QLabel("ph°"))
        self._ph_sl = QSlider(Qt.Horizontal)
        self._ph_sl.setRange(0, 1000)
        self._ph_sl.setValue(_val_to_sl(phase_deg, -180.0, 180.0))
        self._ph_sl.setMinimumWidth(80)
        self._ph_sb = _make_spinbox(-180.0, 180.0, phase_deg, decimals=1, step=1.0)
        row.addWidget(self._ph_sl)
        row.addWidget(self._ph_sb)

        self._amp_sl.valueChanged.connect(self._on_amp_sl)
        self._amp_sb.valueChanged.connect(self._on_amp_sb)
        self._ph_sl.valueChanged.connect(self._on_ph_sl)
        self._ph_sb.valueChanged.connect(self._on_ph_sb)

    def _on_amp_sl(self, pos: int) -> None:
        if self._guard:
            return
        self._guard = True
        self._amp_sb.setValue(_sl_to_val(pos, 0.0, 1.0))
        self._guard = False
        self.changed.emit()

    def _on_amp_sb(self, val: float) -> None:
        if self._guard:
            return
        self._guard = True
        self._amp_sl.setValue(_val_to_sl(val, 0.0, 1.0))
        self._guard = False
        self.changed.emit()

    def _on_ph_sl(self, pos: int) -> None:
        if self._guard:
            return
        self._guard = True
        self._ph_sb.setValue(_sl_to_val(pos, -180.0, 180.0))
        self._guard = False
        self.changed.emit()

    def _on_ph_sb(self, val: float) -> None:
        if self._guard:
            return
        self._guard = True
        self._ph_sl.setValue(_val_to_sl(val, -180.0, 180.0))
        self._guard = False
        self.changed.emit()

    def amp(self) -> float:
        return self._amp_sb.value()

    def phase(self) -> float:
        return self._ph_sb.value()

    def set_values(self, amp: float, phase_deg: float) -> None:
        self._guard = True
        self._amp_sl.setValue(_val_to_sl(amp, 0.0, 1.0))
        self._amp_sb.setValue(amp)
        self._ph_sl.setValue(_val_to_sl(phase_deg, -180.0, 180.0))
        self._ph_sb.setValue(phase_deg)
        self._guard = False


# ── Main window ───────────────────────────────────────────────────────────────

class KickSynthWindow(QWidget):
    """Full KickSynth editor with interactive envelope drag-and-drop."""

    params_changed = Signal()

    _DIAL_LO = 20.0
    _DIAL_HI = 1000.0
    _HIT_RADIUS_PX = 14  # pixels — drag hit radius for knot dots

    def __init__(self, block, block_name: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Window)
        self.block = block
        self.block_name = block_name
        self._guard = False
        self._drag_ax_name: str | None = None   # "pitch" or "amp"
        self._drag_idx: int | None = None

        # Enforce endpoint constraints in case block was loaded from file
        self._enforce_fixed_pts()

        self.setWindowTitle(f"KickSynth — {block_name}")
        self.setMinimumSize(960, 600)

        main = QHBoxLayout(self)
        main.setSpacing(8)

        # ── Left: scrollable controls ─────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(440)

        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setSpacing(6)

        self._build_global_group(ctrl_layout)
        self._build_pitch_group(ctrl_layout)
        self._build_amp_group(ctrl_layout)
        self._build_harmonics_group(ctrl_layout)
        ctrl_layout.addStretch()

        left_scroll.setWidget(controls)
        main.addWidget(left_scroll)

        # ── Right: matplotlib canvas ──────────────────────────────────────
        self._fig = Figure(figsize=(5, 5))
        self._fig.patch.set_facecolor("#1e1e1e")
        self._ax_pitch, self._ax_amp = self._fig.subplots(2, 1)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self._canvas, stretch=1)

        # Wire drag events
        self._canvas.mpl_connect("button_press_event",   self._on_mpl_press)
        self._canvas.mpl_connect("motion_notify_event",  self._on_mpl_motion)
        self._canvas.mpl_connect("button_release_event", self._on_mpl_release)

        self._update_plot()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _enforce_fixed_pts(self) -> None:
        """Pin first knot to t=0 and last knot to t=length."""
        length = self.block.length
        for pts in (self.block._pitch_pts, self.block._amp_pts):
            pts[0][0] = 0.0
            pts[-1][0] = length

    # ── UI builders ───────────────────────────────────────────────────────

    def _build_global_group(self, parent_layout: QVBoxLayout) -> None:
        grp = QGroupBox("Global")
        form = QFormLayout(grp)

        # freq_scale — QDial (rotary knob) + spinbox
        dial_row = QWidget()
        dr = QHBoxLayout(dial_row)
        dr.setContentsMargins(0, 0, 0, 0)
        self._dial = QDial()
        self._dial.setRange(0, 1000)
        self._dial.setNotchesVisible(True)
        self._dial.setFixedSize(64, 64)
        self._dial.setValue(_val_to_sl(self.block.freq_scale,
                                       self._DIAL_LO, self._DIAL_HI, log=True))
        self._fs_sb = _make_spinbox(self._DIAL_LO, self._DIAL_HI,
                                    self.block.freq_scale, decimals=1, step=1.0)
        dr.addWidget(self._dial)
        dr.addWidget(self._fs_sb)
        dr.addStretch()
        form.addRow("Max pitch (Hz)", dial_row)
        self._dial.valueChanged.connect(self._on_dial)
        self._fs_sb.valueChanged.connect(self._on_fs_sb)

        # length — slider + spinbox
        len_row = QWidget()
        lr = QHBoxLayout(len_row)
        lr.setContentsMargins(0, 0, 0, 0)
        self._len_sl = QSlider(Qt.Horizontal)
        self._len_sl.setRange(0, 1000)
        self._len_sl.setMinimumWidth(120)
        self._len_sl.setValue(_val_to_sl(self.block.length, 0.1, 4.0))
        self._len_sb = _make_spinbox(0.1, 4.0, self.block.length,
                                     decimals=2, step=0.05)
        lr.addWidget(self._len_sl)
        lr.addWidget(self._len_sb)
        form.addRow("Length (s)", len_row)
        self._len_sl.valueChanged.connect(self._on_len_sl)
        self._len_sb.valueChanged.connect(self._on_len_sb)

        parent_layout.addWidget(grp)

    def _build_pitch_group(self, parent_layout: QVBoxLayout) -> None:
        grp = QGroupBox("Pitch envelope  (value × max pitch → Hz)")
        layout = QVBoxLayout(grp)
        n = len(self.block._pitch_pts)
        self._pitch_rows: list[_EnvPointRow] = []
        for i, (t, v) in enumerate(self.block._pitch_pts):
            fixed = (i == 0) or (i == n - 1)
            row = _EnvPointRow(i, t, v, (0.0, 4.0), (0.0, 1.0),
                               fixed_t=fixed)
            row.changed.connect(self._on_pitch_changed)
            layout.addWidget(row)
            self._pitch_rows.append(row)
        parent_layout.addWidget(grp)

    def _build_amp_group(self, parent_layout: QVBoxLayout) -> None:
        grp = QGroupBox("Amplitude envelope")
        layout = QVBoxLayout(grp)
        n = len(self.block._amp_pts)
        self._amp_rows: list[_EnvPointRow] = []
        for i, (t, v) in enumerate(self.block._amp_pts):
            fixed = (i == 0) or (i == n - 1)
            row = _EnvPointRow(i, t, v, (0.0, 4.0), (0.0, 1.0),
                               fixed_t=fixed)
            row.changed.connect(self._on_amp_changed)
            layout.addWidget(row)
            self._amp_rows.append(row)
        parent_layout.addWidget(grp)

    def _build_harmonics_group(self, parent_layout: QVBoxLayout) -> None:
        grp = QGroupBox("Harmonics  (2nd … 7th above fundamental)")
        layout = QVBoxLayout(grp)
        self._harm_rows: list[_HarmonicRow] = []
        for i, (amp, phase) in enumerate(self.block._harmonics):
            row = _HarmonicRow(i + 2, amp, phase)
            row.changed.connect(self._on_harm_changed)
            layout.addWidget(row)
            self._harm_rows.append(row)
        parent_layout.addWidget(grp)

    # ── Slots — global controls ───────────────────────────────────────────

    def _on_dial(self, pos: int) -> None:
        if self._guard:
            return
        self._guard = True
        self._fs_sb.setValue(_sl_to_val(pos, self._DIAL_LO, self._DIAL_HI, log=True))
        self._guard = False
        self._flush_global()

    def _on_fs_sb(self, val: float) -> None:
        if self._guard:
            return
        self._guard = True
        self._dial.setValue(_val_to_sl(val, self._DIAL_LO, self._DIAL_HI, log=True))
        self._guard = False
        self._flush_global()

    def _on_len_sl(self, pos: int) -> None:
        if self._guard:
            return
        self._guard = True
        self._len_sb.setValue(_sl_to_val(pos, 0.1, 4.0))
        self._guard = False
        self._flush_global()

    def _on_len_sb(self, val: float) -> None:
        if self._guard:
            return
        self._guard = True
        self._len_sl.setValue(_val_to_sl(val, 0.1, 4.0))
        self._guard = False
        self._flush_global()

    def _flush_global(self) -> None:
        self.block.freq_scale = self._fs_sb.value()
        self.block.length = self._len_sb.value()
        # Enforce last-point times = new length (silently, no changed signal)
        self.block._pitch_pts[-1][0] = self.block.length
        self.block._amp_pts[-1][0] = self.block.length
        self._pitch_rows[-1].set_t(self.block.length)
        self._amp_rows[-1].set_t(self.block.length)
        self._update_plot()
        self.params_changed.emit()

    # ── Slots — envelope spinboxes ────────────────────────────────────────

    def _on_pitch_changed(self) -> None:
        for i, row in enumerate(self._pitch_rows):
            self.block._pitch_pts[i][0] = row.t()
            self.block._pitch_pts[i][1] = row.v()
        # Re-pin endpoints (spinbox is disabled but be defensive)
        self.block._pitch_pts[0][0] = 0.0
        self.block._pitch_pts[-1][0] = self.block.length
        self._update_plot()
        self.params_changed.emit()

    def _on_amp_changed(self) -> None:
        for i, row in enumerate(self._amp_rows):
            self.block._amp_pts[i][0] = row.t()
            self.block._amp_pts[i][1] = row.v()
        self.block._amp_pts[0][0] = 0.0
        self.block._amp_pts[-1][0] = self.block.length
        self._update_plot()
        self.params_changed.emit()

    def _on_harm_changed(self) -> None:
        for i, row in enumerate(self._harm_rows):
            self.block._harmonics[i][0] = row.amp()
            self.block._harmonics[i][1] = row.phase()
        self.params_changed.emit()

    # ── Drag-and-drop on matplotlib canvas ───────────────────────────────

    def _hit_test(self, ax_name: str, px: float, py: float) -> int | None:
        """Return index of the knot dot nearest to display pixel (px, py)."""
        ax = self._ax_pitch if ax_name == "pitch" else self._ax_amp
        pts = self.block._pitch_pts if ax_name == "pitch" else self.block._amp_pts
        y_scale = self.block.freq_scale if ax_name == "pitch" else 1.0

        best_dist = float("inf")
        best_idx: int | None = None
        for i, (t, v) in enumerate(pts):
            x_disp, y_disp = ax.transData.transform((t, v * y_scale))
            d = ((x_disp - px) ** 2 + (y_disp - py) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx if best_dist < self._HIT_RADIUS_PX else None

    def _on_mpl_press(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes == self._ax_pitch:
            ax_name = "pitch"
        elif event.inaxes == self._ax_amp:
            ax_name = "amp"
        else:
            return
        idx = self._hit_test(ax_name, event.x, event.y)
        if idx is not None:
            self._drag_ax_name = ax_name
            self._drag_idx = idx

    def _on_mpl_motion(self, event) -> None:
        if self._drag_idx is None:
            return

        ax = self._ax_pitch if self._drag_ax_name == "pitch" else self._ax_amp
        pts  = self.block._pitch_pts if self._drag_ax_name == "pitch" else self.block._amp_pts
        rows = self._pitch_rows      if self._drag_ax_name == "pitch" else self._amp_rows
        y_scale = self.block.freq_scale if self._drag_ax_name == "pitch" else 1.0
        length = self.block.length
        n = len(pts)

        # Convert display pixels → data coords (works even outside the axes)
        try:
            xd, yd = ax.transData.inverted().transform((event.x, event.y))
        except Exception:
            return

        # Clamp to valid data range
        t_new = float(np.clip(xd, 0.0, length))
        v_new = float(np.clip(yd / y_scale if y_scale else yd, 0.0, 1.0))

        # Pin first and last knots horizontally
        if self._drag_idx == 0:
            t_new = 0.0
        elif self._drag_idx == n - 1:
            t_new = length

        pts[self._drag_idx][0] = t_new
        pts[self._drag_idx][1] = v_new

        # Sync the spinbox row silently (no re-entrant signal)
        rows[self._drag_idx].set_tv(t_new, v_new)

        self._update_plot()
        self.params_changed.emit()

    def _on_mpl_release(self, event) -> None:
        self._drag_ax_name = None
        self._drag_idx = None

    # ── Plot ──────────────────────────────────────────────────────────────

    def _update_plot(self) -> None:
        length = max(self.block.length, 0.01)
        t_plot = np.linspace(0.0, length, 600)

        p_pts = self.block.sorted_pitch_pts()
        freq_env = np.interp(t_plot,
                             [p[0] for p in p_pts],
                             [p[1] for p in p_pts]) * self.block.freq_scale

        a_pts = self.block.sorted_amp_pts()
        amp_env = np.interp(t_plot,
                            [p[0] for p in a_pts],
                            [p[1] for p in a_pts])

        dark   = "#1e1e1e"
        grid_c = "#444444"
        lc     = "#cccccc"

        for ax in (self._ax_pitch, self._ax_amp):
            ax.cla()
            ax.set_facecolor(dark)
            ax.tick_params(colors=lc, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor(grid_c)
            ax.grid(True, color=grid_c, linewidth=0.5)

        # ── Pitch subplot ────────────────────────────────────────────────
        self._ax_pitch.plot(t_plot, freq_env, color="#55aaff", linewidth=1.5)
        self._ax_pitch.scatter(
            [p[0] for p in p_pts],
            [p[1] * self.block.freq_scale for p in p_pts],
            color="#ffaa00", s=70, zorder=5,
        )
        self._ax_pitch.set_xlim(0, length)
        self._ax_pitch.set_ylim(0, self.block.freq_scale * 1.08)
        self._ax_pitch.set_ylabel("Pitch (Hz)", color=lc, fontsize=8)
        self._ax_pitch.set_title(
            "Pitch envelope  —  drag orange dots", color=lc, fontsize=9)

        # ── Amplitude subplot ────────────────────────────────────────────
        self._ax_amp.plot(t_plot, amp_env, color="#88dd88", linewidth=1.5)
        self._ax_amp.scatter(
            [p[0] for p in a_pts],
            [p[1] for p in a_pts],
            color="#ffaa00", s=70, zorder=5,
        )
        self._ax_amp.set_xlim(0, length)
        self._ax_amp.set_ylim(0, 1.08)
        self._ax_amp.set_ylabel("Amplitude", color=lc, fontsize=8)
        self._ax_amp.set_xlabel("Time (s)", color=lc, fontsize=8)
        self._ax_amp.set_title(
            "Amplitude envelope  —  drag orange dots", color=lc, fontsize=9)

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw_idle()

    # ── Refresh from block (e.g. after external load) ─────────────────────

    def refresh(self) -> None:
        self._guard = True
        self._dial.setValue(
            _val_to_sl(self.block.freq_scale, self._DIAL_LO, self._DIAL_HI, log=True))
        self._fs_sb.setValue(self.block.freq_scale)
        self._len_sl.setValue(_val_to_sl(self.block.length, 0.1, 4.0))
        self._len_sb.setValue(self.block.length)
        for i, row in enumerate(self._pitch_rows):
            row.set_tv(*self.block._pitch_pts[i])
        for i, row in enumerate(self._amp_rows):
            row.set_tv(*self.block._amp_pts[i])
        for i, row in enumerate(self._harm_rows):
            row.set_values(*self.block._harmonics[i])
        self._guard = False
        self._update_plot()
