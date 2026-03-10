"""
CompareTab — Tab 2: Compare two concepts algebraically.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSplitter, QFrame
)
from PySide6.QtCore import Qt

from ui.widgets.bit_vector_widget import BitVectorWidget
from ui.widgets.prime_display_widget import PrimeDisplayWidget
from ui.workers.model_worker import TaskWorker


def _tag_label(text: str, object_name: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(object_name)
    return lbl


class _FactorRow(QWidget):
    """Displays a row of factor chip labels."""

    def __init__(self, section_name: str, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        lbl = QLabel(f"{section_name}:")
        lbl.setObjectName("sectionLabel")
        lbl.setFixedWidth(130)
        layout.addWidget(lbl)
        self._chips_layout = QHBoxLayout()
        self._chips_layout.setSpacing(4)
        layout.addLayout(self._chips_layout)
        layout.addStretch()

    def set_factors(self, factors: list[int]):
        # Remove old chips
        while self._chips_layout.count():
            item = self._chips_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for p in factors[:24]:
            chip = QLabel(str(p))
            chip.setObjectName("primeFactors")
            chip.setStyleSheet(
                f"background-color: #313244; border-radius: 4px; padding: 2px 6px; color: {self._color};"
            )
            self._chips_layout.addWidget(chip)
        if len(factors) > 24:
            more = QLabel(f"+{len(factors) - 24}")
            more.setObjectName("statsLabel")
            self._chips_layout.addWidget(more)


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

        lbl_title = QLabel("COMPARE — Comparación algebraica de dos conceptos")
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
        lbl_sim = QLabel("SIMILITUD:")
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

        # ── Factor rows ───────────────────────────────────
        self._row_shared = _FactorRow("COMPARTIDOS (verde)", '#a6e3a1')
        self._row_only_a = _FactorRow("SOLO EN A (amarillo)", '#f9e2af')
        self._row_only_b = _FactorRow("SOLO EN B (durazno)", '#fab387')
        layout.addWidget(self._row_shared)
        layout.addWidget(self._row_only_a)
        layout.addWidget(self._row_only_b)

        # ── LCM composition ───────────────────────────────
        comp_row = QHBoxLayout()
        lbl_lcm = QLabel("Composición A∪B (LCM):")
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
        self._lbl_sim_val.setText("Calculando...")
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
        self._lbl_sub_ab.setText(f"A ⊇ B: {'✓ Sí' if sub_ab else '✗ No'}")
        self._lbl_sub_ba.setText(f"B ⊇ A: {'✓ Sí' if sub_ba else '✗ No'}")
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

    def _on_error(self, msg: str):
        self._lbl_sim_val.setText(f"Error: {msg.split(chr(10))[0]}")
