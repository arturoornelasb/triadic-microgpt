"""
PrimeDisplayWidget — Shows a prime composite number with its factors.

Layout:
  PRIME COMPOSITE (large monospace)
  = 2 × 3 × 7 × 11 × ...
  [n_active / n_total bits active]  [Copiar]
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QClipboard


class PrimeDisplayWidget(QWidget):
    def __init__(self, n_bits: int = 64, parent=None):
        super().__init__(parent)
        self.n_bits = n_bits
        self._composite: int = 0
        self._factors: list[int] = []
        self._n_active: int = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        hdr = QLabel("PRIME COMPOSITE")
        hdr.setObjectName("sectionLabel")
        layout.addWidget(hdr)

        # Composite number
        self._lbl_composite = QLabel("—")
        self._lbl_composite.setObjectName("primeNumber")
        self._lbl_composite.setWordWrap(True)
        self._lbl_composite.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._lbl_composite)

        # Factors
        self._lbl_factors = QLabel("—")
        self._lbl_factors.setObjectName("primeFactors")
        self._lbl_factors.setWordWrap(True)
        layout.addWidget(self._lbl_factors)

        # Bottom row: stats + copy button
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        self._lbl_stats = QLabel("0 / 0 bits activos")
        self._lbl_stats.setObjectName("statsLabel")
        bottom.addWidget(self._lbl_stats)
        bottom.addStretch()
        btn_copy = QPushButton("Copiar Φ")
        btn_copy.setObjectName("smallButton")
        btn_copy.clicked.connect(self._copy)
        bottom.addWidget(btn_copy)
        layout.addLayout(bottom)

    def set_prime(self, composite: int, factors: list[int], n_active: int) -> None:
        self._composite = composite
        self._factors = factors
        self._n_active = n_active
        # Composite display (shorten if huge)
        comp_str = str(composite)
        if len(comp_str) > 40:
            comp_str = comp_str[:20] + '...' + comp_str[-10:]
        self._lbl_composite.setText(f"Φ = {comp_str}")
        # Factors
        if factors:
            factor_str = " × ".join(str(p) for p in factors[:20])
            if len(factors) > 20:
                factor_str += f" × ... ({len(factors)} primos)"
        else:
            factor_str = "(sin factores)"
        self._lbl_factors.setText(f"= {factor_str}")
        self._lbl_stats.setText(f"{n_active} / {self.n_bits} bits activos")

    def reset(self) -> None:
        self._composite = 0
        self._lbl_composite.setText("—")
        self._lbl_factors.setText("—")
        self._lbl_stats.setText(f"0 / {self.n_bits} bits activos")

    def _copy(self):
        if self._composite:
            QApplication.clipboard().setText(str(self._composite))
