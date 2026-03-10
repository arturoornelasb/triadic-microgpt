"""
ExploreTab — Tab 3: N words → similarity heatmap + ranked pairs table.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt

from ui.widgets.mpl_canvas import MplCanvas
from ui.workers.model_worker import TaskWorker


class ExploreTab(QWidget):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._words: list[str] = ['king', 'queen', 'dog', 'cat', 'doctor', 'hospital']
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl_title = QLabel("EXPLORE — Similarity matrix across N concepts")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Word list controls ────────────────────────────
        ctrl_row = QHBoxLayout()
        self._txt_add = QLineEdit()
        self._txt_add.setPlaceholderText("Add word...")
        self._txt_add.setFixedWidth(160)
        self._txt_add.returnPressed.connect(self._add_word)
        ctrl_row.addWidget(self._txt_add)

        btn_add = QPushButton("+ Add")
        btn_add.setObjectName("smallButton")
        btn_add.clicked.connect(self._add_word)
        ctrl_row.addWidget(btn_add)

        btn_del = QPushButton("Remove")
        btn_del.setObjectName("smallButton")
        btn_del.clicked.connect(self._del_word)
        ctrl_row.addWidget(btn_del)

        btn_clear = QPushButton("Clear")
        btn_clear.setObjectName("smallButton")
        btn_clear.clicked.connect(self._clear_words)
        ctrl_row.addWidget(btn_clear)

        ctrl_row.addStretch()

        btn_explore = QPushButton("Explore →")
        btn_explore.setMinimumWidth(110)
        btn_explore.clicked.connect(self._run_explore)
        ctrl_row.addWidget(btn_explore)
        layout.addLayout(ctrl_row)

        # ── Word list ─────────────────────────────────────
        self._word_list = QListWidget()
        self._word_list.setFixedHeight(60)
        self._word_list.setFlow(QListWidget.LeftToRight)
        for w in self._words:
            self._word_list.addItem(w)
        layout.addWidget(self._word_list)

        # ── Status label ──────────────────────────────────
        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statsLabel")
        layout.addWidget(self._lbl_status)

        # ── Main splitter: heatmap | ranked pairs ─────────
        splitter = QSplitter(Qt.Horizontal)

        # Left: heatmap
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 8, 0)
        lbl_hm = QLabel("SIMILARITY MATRIX")
        lbl_hm.setObjectName("sectionLabel")
        left_l.addWidget(lbl_hm)
        self._canvas = MplCanvas(width=5, height=4, dpi=90, tight=False)
        left_l.addWidget(self._canvas)
        splitter.addWidget(left)

        # Right: ranked pairs table
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 0, 0, 0)
        lbl_tbl = QLabel("RANKED PAIRS")
        lbl_tbl.setObjectName("sectionLabel")
        right_l.addWidget(lbl_tbl)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(['A', 'B', 'Similarity', 'Shared'])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setAlternatingRowColors(False)
        right_l.addWidget(self._table)
        splitter.addWidget(right)

        splitter.setSizes([480, 380])
        layout.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------

    def _add_word(self):
        w = self._txt_add.text().strip().lower()
        if w and w not in self._words:
            self._words.append(w)
            self._word_list.addItem(w)
        self._txt_add.clear()

    def _del_word(self):
        row = self._word_list.currentRow()
        if row >= 0:
            self._words.pop(row)
            self._word_list.takeItem(row)

    def _clear_words(self):
        self._words.clear()
        self._word_list.clear()

    def _run_explore(self):
        if len(self._words) < 2:
            self._lbl_status.setText("Add at least 2 words.")
            return
        self._lbl_status.setText("Computing matrix...")
        self._worker = TaskWorker(self._iface.explore, list(self._words))
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        words = result.get('words', self._words)
        matrix = result.get('matrix', [])
        ranked = result.get('pairs', [])

        self._lbl_status.setText(
            f"{len(words)} concepts | {len(ranked)} pairs"
        )
        self._draw_heatmap(words, matrix)
        self._populate_table(ranked)

    def _on_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg.split(chr(10))[0]}")

    def _draw_heatmap(self, words: list, matrix: list):
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')

        if not matrix:
            self._canvas.draw()
            return

        mat = np.array(matrix)
        n = len(words)
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8, color='#cdd6f4')
        ax.set_yticklabels(words, fontsize=8, color='#cdd6f4')
        ax.tick_params(colors='#6c7086')
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475a')

        # Annotate cells — white text with dark outline for readability
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                txt_color = '#1e1e2e' if val > 0.65 else '#f0f0f0'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=7, fontweight='bold', color=txt_color)

        self._canvas.fig.tight_layout(pad=0.3)
        self._canvas.draw()

    def _populate_table(self, ranked: list):
        self._table.setRowCount(0)
        for pair in ranked:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(pair.get('word_a', ''))))
            self._table.setItem(row, 1, QTableWidgetItem(str(pair.get('word_b', ''))))
            sim = pair.get('similarity', 0.0)
            self._table.setItem(row, 2, QTableWidgetItem(f"{sim*100:.1f}%"))
            self._table.setItem(row, 3, QTableWidgetItem(str(pair.get('n_shared', ''))))
