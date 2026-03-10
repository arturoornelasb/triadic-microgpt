"""
ValidateTab — Tab 5: PASS/FAIL semantic validation suite.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QFrame,
    QScrollArea, QHeaderView, QAbstractItemView, QSplitter
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ui.workers.model_worker import TaskWorker


_DEFAULT_GROUPS = {
    'royalty':   ['king', 'queen', 'prince', 'throne'],
    'animals':   ['dog', 'cat', 'fish', 'bird'],
    'medicine':  ['doctor', 'hospital', 'patient', 'medicine'],
    'technology':['computer', 'software', 'internet', 'technology'],
}


class _CheckCard(QFrame):
    """Single check row: [PASS|FAIL] description: value"""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        self._badge = QLabel("—")
        self._badge.setFixedWidth(50)
        layout.addWidget(self._badge)
        lbl = QLabel(label)
        lbl.setObjectName("statsLabel")
        layout.addWidget(lbl)
        layout.addStretch()
        self._val = QLabel("")
        layout.addWidget(self._val)

    def set_result(self, passed: bool, value_text: str):
        self.setObjectName("passCard" if passed else "failCard")
        self._badge.setObjectName("passLabel" if passed else "failLabel")
        self._badge.setText("PASS ✓" if passed else "FAIL ✗")
        self._val.setText(value_text)
        # Force style refresh
        for w in (self, self._badge):
            w.style().unpolish(w)
            w.style().polish(w)

    def reset(self):
        self.setObjectName("card")
        self._badge.setObjectName("")
        self._badge.setText("—")
        self._val.setText("")
        self.style().unpolish(self)
        self.style().polish(self)


class ValidateTab(QWidget):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._groups: dict[str, list[str]] = dict(_DEFAULT_GROUPS)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl_title = QLabel("VALIDATE — Semantic quality audit")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Group editor ──────────────────────────────────
        editor_frame = QFrame()
        editor_frame.setObjectName("card")
        editor_l = QVBoxLayout(editor_frame)
        editor_l.setContentsMargins(10, 8, 10, 8)

        lbl_groups = QLabel("WORD GROUPS")
        lbl_groups.setObjectName("sectionLabel")
        editor_l.addWidget(lbl_groups)

        # Add group row
        add_row = QHBoxLayout()
        self._txt_group_name = QLineEdit()
        self._txt_group_name.setPlaceholderText("Group name...")
        self._txt_group_name.setFixedWidth(140)
        add_row.addWidget(self._txt_group_name)
        self._txt_group_words = QLineEdit()
        self._txt_group_words.setPlaceholderText("comma-separated words...")
        add_row.addWidget(self._txt_group_words)
        btn_add_group = QPushButton("+ Group")
        btn_add_group.setObjectName("smallButton")
        btn_add_group.clicked.connect(self._add_group)
        add_row.addWidget(btn_add_group)
        btn_reset = QPushButton("Reset")
        btn_reset.setObjectName("smallButton")
        btn_reset.clicked.connect(self._reset_groups)
        add_row.addWidget(btn_reset)
        editor_l.addLayout(add_row)

        # Groups table
        self._tbl_groups = QTableWidget(0, 3)
        self._tbl_groups.setHorizontalHeaderLabels(['Group', 'Words', ''])
        self._tbl_groups.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._tbl_groups.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tbl_groups.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._tbl_groups.setFixedHeight(130)
        self._tbl_groups.setEditTriggers(QAbstractItemView.NoEditTriggers)
        editor_l.addWidget(self._tbl_groups)
        layout.addWidget(editor_frame)

        self._populate_groups_table()

        # ── Run button ────────────────────────────────────
        run_row = QHBoxLayout()
        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statsLabel")
        run_row.addWidget(self._lbl_status)
        run_row.addStretch()
        btn_run = QPushButton("Run Validation →")
        btn_run.setMinimumWidth(150)
        btn_run.clicked.connect(self._run_validate)
        run_row.addWidget(btn_run)
        layout.addLayout(run_row)

        # ── Global result banner ──────────────────────────
        self._lbl_global = QLabel("—")
        self._lbl_global.setObjectName("titleLabel")
        self._lbl_global.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_global)

        # ── Check cards ───────────────────────────────────
        self._card_diversity = _CheckCard("Diversity: unique signatures")
        self._card_bits = _CheckCard("Active Bits: mean active bits")
        self._card_ordering = _CheckCard("Semantic Ordering: intra > inter gap")
        for c in (self._card_diversity, self._card_bits, self._card_ordering):
            layout.addWidget(c)

        # ── Per-group table ───────────────────────────────
        lbl_per = QLabel("PER-GROUP RESULTS")
        lbl_per.setObjectName("sectionLabel")
        layout.addWidget(lbl_per)

        self._tbl_results = QTableWidget(0, 5)
        self._tbl_results.setHorizontalHeaderLabels(
            ['Group', 'Intra', 'Inter', 'Gap', 'Status']
        )
        self._tbl_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._tbl_results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self._tbl_results, stretch=1)

    # ------------------------------------------------------------------

    def _populate_groups_table(self):
        self._tbl_groups.setRowCount(0)
        for name, words in self._groups.items():
            row = self._tbl_groups.rowCount()
            self._tbl_groups.insertRow(row)
            self._tbl_groups.setItem(row, 0, QTableWidgetItem(name))
            self._tbl_groups.setItem(row, 1, QTableWidgetItem(', '.join(words)))
            btn = QPushButton("✕")
            btn.setObjectName("dangerButton")
            btn.setFixedWidth(32)
            btn.clicked.connect(lambda _, n=name: self._remove_group(n))
            self._tbl_groups.setCellWidget(row, 2, btn)

    def _add_group(self):
        name = self._txt_group_name.text().strip()
        words_str = self._txt_group_words.text().strip()
        if not name or not words_str:
            return
        words = [w.strip() for w in words_str.split(',') if w.strip()]
        self._groups[name] = words
        self._txt_group_name.clear()
        self._txt_group_words.clear()
        self._populate_groups_table()

    def _remove_group(self, name: str):
        self._groups.pop(name, None)
        self._populate_groups_table()

    def _reset_groups(self):
        self._groups = dict(_DEFAULT_GROUPS)
        self._populate_groups_table()

    def _run_validate(self):
        if not self._groups:
            self._lbl_status.setText("Add at least one group.")
            return
        self._lbl_status.setText("Running validation...")
        self._worker = TaskWorker(self._iface.validate, dict(self._groups))
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        passed_all = result.get('overall_pass', False)
        checks = result.get('checks', {})
        diversity = checks.get('diversity', {})
        active_bits = checks.get('active_bits', {})
        ordering = checks.get('semantic_ordering', {})
        group_details = result.get('group_details', {})
        n_unique = result.get('unique_signatures', 0)
        n_total = result.get('n_concepts', 0)

        self._lbl_global.setText("✓ PASS — Model passes all checks" if passed_all
                                 else "✗ FAIL — Some checks failed")
        self._lbl_global.setObjectName("passLabel" if passed_all else "failLabel")
        self._lbl_global.style().unpolish(self._lbl_global)
        self._lbl_global.style().polish(self._lbl_global)

        # Diversity
        self._card_diversity.set_result(
            diversity.get('pass', False),
            f"{n_unique}/{n_total} unique signatures"
        )

        # Active bits — value is a ratio (0-1)
        n_bits = self._iface.n_bits
        avg_ratio = active_bits.get('value', 0.0)
        mean_active = avg_ratio * n_bits
        self._card_bits.set_result(
            active_bits.get('pass', False),
            f"{mean_active:.1f}/{n_bits} active bits ({avg_ratio*100:.0f}%)"
        )

        # Semantic ordering
        gap = ordering.get('gap', 0.0)
        detail = ordering.get('detail', '')
        self._card_ordering.set_result(
            ordering.get('pass', False),
            f"gap={gap:+.3f} | {detail}"
        )

        # Per-group table
        self._tbl_results.setRowCount(0)
        for group_name, grp in group_details.items():
            row = self._tbl_results.rowCount()
            self._tbl_results.insertRow(row)
            passed = grp.get('pass', False)
            self._tbl_results.setItem(row, 0, QTableWidgetItem(group_name))
            self._tbl_results.setItem(row, 1, QTableWidgetItem(f"{grp.get('intra_sim', 0)*100:.1f}%"))
            self._tbl_results.setItem(row, 2, QTableWidgetItem(f"{grp.get('inter_sim', 0)*100:.1f}%"))
            self._tbl_results.setItem(row, 3, QTableWidgetItem(f"{grp.get('gap', 0):+.3f}"))
            status_item = QTableWidgetItem('PASS ✓' if passed else 'FAIL ✗')
            status_item.setForeground(QColor('#a6e3a1' if passed else '#f38ba8'))
            self._tbl_results.setItem(row, 4, status_item)

        self._lbl_status.setText(
            f"Validation complete — {'PASS' if passed_all else 'FAIL'} | {n_total} concepts"
        )

    def _on_error(self, msg: str):
        self._lbl_status.setText(f"Error: {msg.split(chr(10))[0]}")
