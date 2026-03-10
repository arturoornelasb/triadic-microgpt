"""
CompareTab — Tab 2: Compare two concepts algebraically.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSplitter, QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal
from PySide6.QtWidgets import QLayout

from ui.widgets.bit_vector_widget import BitVectorWidget
from ui.workers.model_worker import TaskWorker


class _FlowLayout(QLayout):
    """Simple flow layout that wraps items to next line when row is full."""

    def __init__(self, parent=None, margin=0, spacing=4):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def _do_layout(self, rect, test_only=False):
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            sz = item.sizeHint()
            next_x = x + sz.width() + self._spacing
            if next_x - self._spacing > effective.right() + 1 and line_height > 0:
                x = effective.x()
                y += line_height + self._spacing
                next_x = x + sz.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), sz))
            x = next_x
            line_height = max(line_height, sz.height())

        return y + line_height - rect.y() + m.bottom()


class _FactorRow(QWidget):
    """Displays a wrapping row of clickable factor chips."""
    prime_clicked = Signal(int)  # emits the prime number when a chip is clicked

    def __init__(self, section_name: str, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        lbl = QLabel(f"{section_name}:")
        lbl.setObjectName("sectionLabel")
        outer.addWidget(lbl)

        self._chips_widget = QWidget()
        self._chips_layout = _FlowLayout(self._chips_widget, margin=0, spacing=4)
        outer.addWidget(self._chips_widget)

    def set_factors(self, factors: list[int]):
        # Remove old chips
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for p in factors[:20]:
            chip = QPushButton(str(p))
            chip.setObjectName("primeFactors")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton {{ background-color: #313244; border-radius: 4px; padding: 2px 6px; "
                f"color: {self._color}; border: 1px solid transparent; }}"
                f"QPushButton:hover {{ border: 1px solid {self._color}; background-color: #45475a; }}"
            )
            chip.clicked.connect(lambda checked, prime=p: self.prime_clicked.emit(prime))
            self._chips_layout.addWidget(chip)
        if len(factors) > 20:
            more = QLabel(f"+{len(factors) - 20}")
            more.setObjectName("statsLabel")
            self._chips_layout.addWidget(more)
        self._chips_widget.updateGeometry()


class CompareTab(QWidget):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl_title = QLabel("COMPARE — Algebraic comparison of two concepts")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Input row ─────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("A:"))
        self._txt_a = QLineEdit()
        self._txt_a.setPlaceholderText("e.g. king")
        row.addWidget(self._txt_a)
        row.addWidget(QLabel("B:"))
        self._txt_b = QLineEdit()
        self._txt_b.setPlaceholderText("e.g. queen")
        row.addWidget(self._txt_b)
        btn = QPushButton("Compare →")
        btn.setMinimumWidth(110)
        btn.clicked.connect(self._run_compare)
        row.addWidget(btn)
        layout.addLayout(row)

        # ── Similarity bar ────────────────────────────────
        sim_row = QHBoxLayout()
        lbl_sim = QLabel("SIMILARITY:")
        lbl_sim.setObjectName("sectionLabel")
        lbl_sim.setFixedWidth(90)
        sim_row.addWidget(lbl_sim)
        self._lbl_sim_val = QLabel("—")
        self._lbl_sim_val.setObjectName("statusLabel")
        sim_row.addWidget(self._lbl_sim_val)
        sim_row.addStretch()
        layout.addLayout(sim_row)

        # ── Subsumption ───────────────────────────────────
        sub_row = QHBoxLayout()
        self._lbl_sub_ab = QLabel("A ⊇ B: —")
        self._lbl_sub_ab.setObjectName("statsLabel")
        sub_row.addWidget(self._lbl_sub_ab)
        sub_row.addSpacing(20)
        self._lbl_sub_ba = QLabel("B ⊇ A: —")
        self._lbl_sub_ba.setObjectName("statsLabel")
        sub_row.addWidget(self._lbl_sub_ba)
        sub_row.addStretch()
        layout.addLayout(sub_row)

        # ── Factor rows (clickable) ──────────────────────
        self._row_shared = _FactorRow("SHARED (green) -- click to inspect", '#a6e3a1')
        self._row_only_a = _FactorRow("ONLY IN A (yellow) -- click to inspect", '#f9e2af')
        self._row_only_b = _FactorRow("ONLY IN B (peach) -- click to inspect", '#fab387')
        for row in (self._row_shared, self._row_only_a, self._row_only_b):
            row.prime_clicked.connect(self._inspect_prime)
        layout.addWidget(self._row_shared)
        layout.addWidget(self._row_only_a)
        layout.addWidget(self._row_only_b)

        # ── LCM composition ───────────────────────────────
        comp_row = QHBoxLayout()
        lbl_lcm = QLabel("Composition A∪B (LCM):")
        lbl_lcm.setObjectName("sectionLabel")
        comp_row.addWidget(lbl_lcm)
        self._lbl_lcm = QLabel("—")
        self._lbl_lcm.setObjectName("primeNumber")
        comp_row.addWidget(self._lbl_lcm)
        comp_row.addStretch()
        layout.addLayout(comp_row)

        # ── Bit vectors side by side ──────────────────────
        lbl_bv = QLabel("BIT VECTORS")
        lbl_bv.setObjectName("sectionLabel")
        layout.addWidget(lbl_bv)

        bv_frame = QFrame()
        bv_layout = QVBoxLayout(bv_frame)
        bv_layout.setSpacing(4)

        a_row = QHBoxLayout()
        lbl_a = QLabel("A:")
        lbl_a.setFixedWidth(20)
        a_row.addWidget(lbl_a)
        self._bv_a = BitVectorWidget(n_bits=self._iface.n_bits)
        a_row.addWidget(self._bv_a)
        bv_layout.addLayout(a_row)

        b_row = QHBoxLayout()
        lbl_b = QLabel("B:")
        lbl_b.setFixedWidth(20)
        b_row.addWidget(lbl_b)
        self._bv_b = BitVectorWidget(n_bits=self._iface.n_bits)
        b_row.addWidget(self._bv_b)
        bv_layout.addLayout(b_row)

        layout.addWidget(bv_frame)
        layout.addStretch()

    # ------------------------------------------------------------------

    def _run_compare(self):
        a = self._txt_a.text().strip()
        b = self._txt_b.text().strip()
        if not a or not b:
            return
        self._lbl_sim_val.setText("Computing...")
        self._worker = TaskWorker(self._iface.compare, a, b)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        sim = result.get('similarity', 0.0)
        shared = result.get('shared_factors', [])
        only_a = result.get('only_a_factors', [])
        only_b = result.get('only_b_factors', [])
        sub_ab = result.get('a_subsumes_b', False)
        sub_ba = result.get('b_subsumes_a', False)
        composition = result.get('composition', 0)
        bits_a = result.get('enc_a', {}).get('bits', [])
        bits_b = result.get('enc_b', {}).get('bits', [])

        pct = int(sim * 100)
        self._lbl_sim_val.setText(f"{pct}%  {'▓' * (pct // 5)}{'░' * (20 - pct // 5)}")
        self._lbl_sub_ab.setText(f"A ⊇ B: {'✓ Yes' if sub_ab else '✗ No'}")
        self._lbl_sub_ba.setText(f"B ⊇ A: {'✓ Yes' if sub_ba else '✗ No'}")
        self._row_shared.set_factors(shared)
        self._row_only_a.set_factors(only_a)
        self._row_only_b.set_factors(only_b)

        comp_str = str(composition)
        if len(comp_str) > 30:
            comp_str = comp_str[:15] + '...' + comp_str[-8:]
        self._lbl_lcm.setText(comp_str if composition else "—")

        if bits_a and bits_b:
            self._bv_a.set_compare(bits_a, bits_b)
            self._bv_b.set_compare(bits_b, bits_a)

    def _inspect_prime(self, prime: int):
        from ui.widgets.prime_inspector_dialog import PrimeInspectorDialog
        dlg = PrimeInspectorDialog(self._iface, prime, parent=self)
        dlg.exec()

    def _on_error(self, msg: str):
        self._lbl_sim_val.setText(f"Error: {msg.split(chr(10))[0]}")
