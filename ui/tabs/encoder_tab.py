"""
EncoderTab — Tab 1: Text → bits → prime signature.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QSplitter, QFrame, QScrollArea
)
from PySide6.QtCore import Qt

from ui.widgets.bit_vector_widget import BitVectorWidget
from ui.widgets.prime_display_widget import PrimeDisplayWidget
from ui.widgets.mpl_canvas import MplCanvas
from ui.workers.model_worker import TaskWorker


class EncoderTab(QWidget):
    """
    Layout:
      [Input text area]  [Encode →]
      BIT VECTOR: 64 colored squares
      PRIME COMPOSITE: Φ = ...
      PROJECTION: bar chart
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Section title ─────────────────────────────────
        lbl_title = QLabel("ENCODER — Texto → Prime Signature")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Input row ─────────────────────────────────────
        input_row = QHBoxLayout()
        self._txt_input = QTextEdit()
        self._txt_input.setPlaceholderText("Escribe un concepto o texto (e.g. 'king', 'medicine', 'love')…")
        self._txt_input.setFixedHeight(70)
        input_row.addWidget(self._txt_input)

        btn_encode = QPushButton("Encode →")
        btn_encode.setMinimumWidth(100)
        btn_encode.clicked.connect(self._run_encode)
        input_row.addWidget(btn_encode, alignment=Qt.AlignTop)
        layout.addLayout(input_row)

        # ── Status label ──────────────────────────────────
        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statsLabel")
        layout.addWidget(self._lbl_status)

        # ── Splitter: left=bit+prime, right=chart ─────────
        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # Bit vector section
        lbl_bits = QLabel("BIT VECTOR")
        lbl_bits.setObjectName("sectionLabel")
        left_layout.addWidget(lbl_bits)

        self._bit_widget = BitVectorWidget(n_bits=self._iface.n_bits)
        left_layout.addWidget(self._bit_widget)

        # Prime display section
        lbl_prime = QLabel("PRIME COMPOSITE")
        lbl_prime.setObjectName("sectionLabel")
        left_layout.addWidget(lbl_prime)

        self._prime_widget = PrimeDisplayWidget(n_bits=self._iface.n_bits)
        left_layout.addWidget(self._prime_widget)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right panel — projection chart
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)

        lbl_proj = QLabel("PROJECTION (tanh activations per bit)")
        lbl_proj.setObjectName("sectionLabel")
        right_layout.addWidget(lbl_proj)

        self._canvas = MplCanvas(width=5, height=4, dpi=90)
        right_layout.addWidget(self._canvas)

        splitter.addWidget(right)
        splitter.setSizes([400, 500])
        layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------

    def _run_encode(self):
        text = self._txt_input.toPlainText().strip()
        if not text:
            return
        self._lbl_status.setText("Codificando...")
        self._worker = TaskWorker(self._iface.encode, text)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        bits = result.get('bits', [])
        composite = result.get('composite', 0)
        factors = result.get('factors', [])
        n_active = result.get('n_active', 0)
        projection = result.get('projection', [])

        self._bit_widget.set_bits(bits, projections=projection if len(projection) else None)
        self._prime_widget.set_prime(composite, factors, n_active)
        self._lbl_status.setText(
            f"{n_active}/{self._iface.n_bits} bits activos | {len(factors)} factores primos"
        )
        self._draw_projection(bits, projection)

    def _on_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg}")

    def _draw_projection(self, bits: list, projection: list):
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')

        n = len(projection)
        if n == 0:
            self._canvas.draw()
            return

        colors = ['#a6e3a1' if b else '#f38ba8' for b in bits]
        x = np.arange(n)
        ax.bar(x, projection, color=colors, width=0.8)
        ax.axhline(0, color='#6c7086', linewidth=0.8, linestyle='--')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Bit index", color='#cdd6f4', fontsize=9)
        ax.set_ylabel("tanh activation", color='#cdd6f4', fontsize=9)
        ax.tick_params(colors='#6c7086', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475a')
        self._canvas.fig.tight_layout(pad=0.5)
        self._canvas.draw()
