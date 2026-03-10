"""
AnalogyTab — Tab 4: A:B::C:? prime algebra reasoning.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QFrame,
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt

from ui.workers.model_worker import TaskWorker


class AnalogyTab(QWidget):
    """
    Layout:
      A: [  ]  es a  B: [  ]  como  C: [  ]  es a  ?
      [Calcular →]

      TRANSFORMACIÓN A→B: eliminados [...] añadidos [...]
      TARGET Φ = ...
      RESULTADO:
        Top-1: word (sim%)  ✓/✗
        ...
      VERIFICACIÓN: sim% | ≥ mediana: ✓/✗
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        lbl_title = QLabel("ANALOGY — A:B::C:? algebraic reasoning")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Input row ─────────────────────────────────────
        input_frame = QFrame()
        input_frame.setObjectName("card")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(12, 10, 12, 10)
        input_layout.setSpacing(8)

        def _make_input(placeholder):
            t = QLineEdit()
            t.setPlaceholderText(placeholder)
            t.setFixedWidth(120)
            return t

        self._txt_a = _make_input("e.g. king")
        self._txt_b = _make_input("e.g. queen")
        self._txt_c = _make_input("e.g. man")

        input_layout.addWidget(QLabel("A:"))
        input_layout.addWidget(self._txt_a)
        input_layout.addWidget(QLabel("is to"))
        input_layout.addWidget(QLabel("B:"))
        input_layout.addWidget(self._txt_b)
        input_layout.addWidget(QLabel("as"))
        input_layout.addWidget(QLabel("C:"))
        input_layout.addWidget(self._txt_c)
        input_layout.addWidget(QLabel("is to"))
        lbl_q = QLabel("?")
        lbl_q.setObjectName("titleLabel")
        input_layout.addWidget(lbl_q)
        input_layout.addStretch()

        btn = QPushButton("Compute →")
        btn.setMinimumWidth(120)
        btn.clicked.connect(self._run_analogy)
        input_layout.addWidget(btn)
        layout.addWidget(input_frame)

        # ── Status ────────────────────────────────────────
        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statsLabel")
        layout.addWidget(self._lbl_status)

        # ── Transform display ─────────────────────────────
        trans_frame = QFrame()
        trans_frame.setObjectName("card")
        trans_layout = QVBoxLayout(trans_frame)
        trans_layout.setContentsMargins(12, 10, 12, 10)

        lbl_trans = QLabel("TRANSFORMATION A→B")
        lbl_trans.setObjectName("sectionLabel")
        trans_layout.addWidget(lbl_trans)

        removed_row = QHBoxLayout()
        removed_row.addWidget(QLabel("Removed:"))
        self._lbl_removed = QLabel("—")
        self._lbl_removed.setObjectName("primeFactors")
        self._lbl_removed.setStyleSheet("color: #f38ba8;")
        removed_row.addWidget(self._lbl_removed)
        removed_row.addStretch()
        trans_layout.addLayout(removed_row)

        added_row = QHBoxLayout()
        added_row.addWidget(QLabel("Added:"))
        self._lbl_added = QLabel("—")
        self._lbl_added.setObjectName("primeFactors")
        self._lbl_added.setStyleSheet("color: #a6e3a1;")
        added_row.addWidget(self._lbl_added)
        added_row.addStretch()
        trans_layout.addLayout(added_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("TARGET Φ:"))
        self._lbl_target = QLabel("—")
        self._lbl_target.setObjectName("primeNumber")
        target_row.addWidget(self._lbl_target)
        target_row.addStretch()
        trans_layout.addLayout(target_row)
        layout.addWidget(trans_frame)

        # ── Results table ─────────────────────────────────
        lbl_results = QLabel("RESULTS (vocabulary matches)")
        lbl_results.setObjectName("sectionLabel")
        layout.addWidget(lbl_results)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(['Rank', 'Word', 'Similarity', 'Verif.'])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setFixedHeight(160)
        layout.addWidget(self._table)

        # ── Verification summary ──────────────────────────
        verif_row = QHBoxLayout()
        lbl_verif = QLabel("VERIFICATION:")
        lbl_verif.setObjectName("sectionLabel")
        verif_row.addWidget(lbl_verif)
        self._lbl_verif = QLabel("—")
        verif_row.addWidget(self._lbl_verif)
        verif_row.addStretch()
        layout.addLayout(verif_row)

        layout.addStretch()

    # ------------------------------------------------------------------

    def _run_analogy(self):
        a = self._txt_a.text().strip()
        b = self._txt_b.text().strip()
        c = self._txt_c.text().strip()
        if not all([a, b, c]):
            return
        self._lbl_status.setText("Computing analogy...")
        self._worker = TaskWorker(self._iface.analogy, a, b, c)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        removed = result.get('transform_removed', [])
        added = result.get('transform_added', [])
        target = result.get('target_prime', 0)
        matches = result.get('matches', [])
        median_sim = result.get('median_sim', 0.5)
        top_sim = matches[0]['similarity'] if matches else 0.0
        verified = top_sim >= median_sim

        self._lbl_removed.setText(
            ' '.join(str(p) for p in removed[:16]) if removed else '(none)'
        )
        self._lbl_added.setText(
            ' '.join(str(p) for p in added[:16]) if added else '(none)'
        )
        t_str = str(target)
        if len(t_str) > 30:
            t_str = t_str[:15] + '...' + t_str[-8:]
        self._lbl_target.setText(t_str if target else '—')

        self._table.setRowCount(0)
        for i, m in enumerate(matches[:10]):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            self._table.setItem(row, 1, QTableWidgetItem(m.get('word', '')))
            sim = m.get('similarity', 0.0)
            self._table.setItem(row, 2, QTableWidgetItem(f"{sim*100:.1f}%"))
            v = m.get('similarity', 0.0) >= median_sim
            item = QTableWidgetItem('✓' if v else '✗')
            item.setForeground(
                __import__('PySide6.QtGui', fromlist=['QColor']).QColor(
                    '#a6e3a1' if v else '#f38ba8'
                )
            )
            self._table.setItem(row, 3, item)

        verif_str = (
            f"Top-1 sim={top_sim*100:.1f}% | median={median_sim*100:.1f}% | "
            f"{'✓ VERIFIED' if verified else '✗ NOT verified'}"
        )
        self._lbl_verif.setObjectName('passLabel' if verified else 'failLabel')
        self._lbl_verif.setText(verif_str)
        self._lbl_verif.style().unpolish(self._lbl_verif)
        self._lbl_verif.style().polish(self._lbl_verif)
        self._lbl_status.setText(f"{len(matches)} candidates found")

    def _on_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg.split(chr(10))[0]}")
