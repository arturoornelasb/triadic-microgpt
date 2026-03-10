"""
BenchmarksTab — Tab 7: Browse and visualize stored benchmark JSON results.
Works without a loaded model.
"""
import json
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QTableWidget, QTableWidgetItem, QSplitter, QFrame,
    QPushButton, QHeaderView, QAbstractItemView, QLineEdit,
    QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ui.widgets.mpl_canvas import MplCanvas


class _MetricCard(QFrame):
    """Small card showing a key metric."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedWidth(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("sectionLabel")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        self._val = QLabel("—")
        self._val.setObjectName("titleLabel")
        self._val.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._val)

    def set_value(self, text: str):
        self._val.setText(text)


class BenchmarksTab(QWidget):
    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self._project_root = project_root
        self._results_dir = project_root / 'benchmarks' / 'results'
        self._current_data: dict | None = None
        self._setup_ui()
        self._refresh_file_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        lbl_title = QLabel("BENCHMARKS — JSON results browser")
        lbl_title.setObjectName("sectionLabel")
        layout.addWidget(lbl_title)

        # ── Main splitter ─────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # LEFT: file list
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 8, 0)
        left_l.setSpacing(6)

        search_row = QHBoxLayout()
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("Filtrar archivos...")
        self._txt_filter.textChanged.connect(self._filter_files)
        search_row.addWidget(self._txt_filter)
        btn_refresh = QPushButton("↻")
        btn_refresh.setObjectName("smallButton")
        btn_refresh.setFixedWidth(32)
        btn_refresh.clicked.connect(self._refresh_file_list)
        search_row.addWidget(btn_refresh)
        left_l.addLayout(search_row)

        self._file_list = QListWidget()
        self._file_list.currentRowChanged.connect(self._on_file_selected)
        left_l.addWidget(self._file_list)
        splitter.addWidget(left)

        # RIGHT: results view
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 0, 0, 0)
        right_l.setSpacing(8)

        # Metadata row
        self._lbl_meta = QLabel("")
        self._lbl_meta.setObjectName("statsLabel")
        self._lbl_meta.setWordWrap(True)
        right_l.addWidget(self._lbl_meta)

        # Metric cards row
        cards_row = QHBoxLayout()
        self._cards: list[_MetricCard] = []
        for label in ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4']:
            card = _MetricCard(label)
            self._cards.append(card)
            cards_row.addWidget(card)
        cards_row.addStretch()
        right_l.addLayout(cards_row)

        # Details table
        lbl_details = QLabel("DETAILS")
        lbl_details.setObjectName("sectionLabel")
        right_l.addWidget(lbl_details)

        self._tbl_details = QTableWidget(0, 1)
        self._tbl_details.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl_details.setSelectionBehavior(QAbstractItemView.SelectRows)
        right_l.addWidget(self._tbl_details, stretch=1)

        # Chart
        lbl_chart = QLabel("CHART")
        lbl_chart.setObjectName("sectionLabel")
        right_l.addWidget(lbl_chart)
        self._canvas = MplCanvas(width=6, height=2.5, dpi=90)
        right_l.addWidget(self._canvas)

        splitter.addWidget(right)
        splitter.setSizes([260, 700])
        layout.addWidget(splitter, stretch=1)

        self._all_files: list[str] = []

    # ------------------------------------------------------------------

    def _refresh_file_list(self):
        self._all_files = []
        if self._results_dir.exists():
            self._all_files = sorted(
                f.name for f in self._results_dir.glob('*.json')
            )
        self._render_file_list(self._all_files)

    def _filter_files(self, text: str):
        filtered = [f for f in self._all_files if text.lower() in f.lower()]
        self._render_file_list(filtered)

    def _render_file_list(self, files: list[str]):
        self._file_list.clear()
        for f in files:
            self._file_list.addItem(f)

    def _on_file_selected(self, row: int):
        if row < 0:
            return
        item = self._file_list.item(row)
        if not item:
            return
        path = self._results_dir / item.text()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception as e:
            self._lbl_meta.setText(f"Error leyendo archivo: {e}")
            return
        self._current_data = data
        self._display_data(data)

    def _display_data(self, data: dict):
        # Metadata
        benchmark = data.get('benchmark', '—')
        version = data.get('version', '—')
        date = data.get('date', '—')
        model = data.get('model_checkpoint', '—')
        self._lbl_meta.setText(
            f"Benchmark: {benchmark} | Version: {version} | Date: {date} | Model: {Path(model).name if model else '—'}"
        )

        metrics = data.get('metrics', {})
        details = data.get('details', [])

        # Dispatch by benchmark type
        btype = benchmark.lower() if isinstance(benchmark, str) else ''
        if 'analogy' in btype:
            self._display_analogy(metrics, details)
        elif 'bit_entropy' in btype or 'entropy' in btype:
            self._display_entropy(metrics, data)
        elif 'language' in btype:
            self._display_language(metrics, details)
        elif 'subsumption' in btype:
            self._display_subsumption(metrics, details)
        elif 'comparison' in btype or 'geometric' in btype:
            self._display_generic(metrics, details)
        else:
            self._display_generic(metrics, details)

    # ── Analogy display ────────────────────────────────────────────────

    def _display_analogy(self, metrics: dict, details: list):
        self._update_cards([
            ('Top-1 Acc', f"{metrics.get('top1_accuracy', 0)*100:.1f}%"),
            ('Top-5 Acc', f"{metrics.get('top5_accuracy', 0)*100:.1f}%"),
            ('Verif. Acc', f"{metrics.get('verification_accuracy', 0)*100:.1f}%"),
            ('N Analogies', str(len(details))),
        ])

        headers = ['Analogy', 'Correct', 'Top-1', 'Top-5', 'Verif.']
        self._tbl_details.setColumnCount(len(headers))
        self._tbl_details.setHorizontalHeaderLabels(headers)
        self._tbl_details.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tbl_details.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tbl_details.setRowCount(0)
        for d in details:
            row = self._tbl_details.rowCount()
            self._tbl_details.insertRow(row)
            self._tbl_details.setItem(row, 0, QTableWidgetItem(d.get('analogy', '')))
            self._tbl_details.setItem(row, 1, QTableWidgetItem(d.get('correct_answer', '')))
            self._tbl_details.setItem(row, 2, QTableWidgetItem(str(d.get('top1', ''))))
            top5 = d.get('top5', [])
            self._tbl_details.setItem(row, 3, QTableWidgetItem(', '.join(top5[:3]) if isinstance(top5, list) else str(top5)))
            for col_idx, key in [(4, 'above_median')]:
                val = d.get(key, False)
                item = QTableWidgetItem('✓' if val else '✗')
                item.setForeground(QColor('#a6e3a1' if val else '#f38ba8'))
                self._tbl_details.setItem(row, col_idx, item)

        # Bar chart: top1 vs top5 vs verif
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')
        labels = ['Top-1', 'Top-5', 'Verif.']
        vals = [
            metrics.get('top1_accuracy', 0) * 100,
            metrics.get('top5_accuracy', 0) * 100,
            metrics.get('verification_accuracy', 0) * 100,
        ]
        colors = ['#89b4fa', '#a6e3a1', '#f9e2af']
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylim(0, 105)
        ax.set_ylabel('%', color='#cdd6f4', fontsize=9)
        ax.tick_params(colors='#cdd6f4', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475a')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', color='#cdd6f4', fontsize=8)
        self._canvas.fig.tight_layout(pad=0.5)
        self._canvas.draw()

    # ── Entropy display ────────────────────────────────────────────────

    def _display_entropy(self, metrics: dict, data: dict):
        mean_h = metrics.get('mean_entropy', 0.0)
        dead = metrics.get('n_dead_bits', 0)
        n_bits = data.get('model_config', {}).get('n_bits', 64) if isinstance(data.get('model_config'), dict) else 64
        self._update_cards([
            ('Mean Entropy', f'{mean_h:.3f}'),
            ('Dead Bits', f'{dead}/{n_bits}'),
            ('Concepts', str(data.get('num_concepts', '—'))),
            ('Bits', str(n_bits)),
        ])

        self._tbl_details.setColumnCount(2)
        self._tbl_details.setHorizontalHeaderLabels(['Metric', 'Value'])
        self._tbl_details.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._tbl_details.setRowCount(0)
        for k, v in metrics.items():
            row = self._tbl_details.rowCount()
            self._tbl_details.insertRow(row)
            self._tbl_details.setItem(row, 0, QTableWidgetItem(str(k)))
            self._tbl_details.setItem(row, 1, QTableWidgetItem(f"{v:.4f}" if isinstance(v, float) else str(v)))

        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')
        per_bit = metrics.get('per_bit_entropy', [])
        if per_bit:
            x = np.arange(len(per_bit))
            colors = ['#f38ba8' if h < 0.3 else '#89b4fa' for h in per_bit]
            ax.bar(x, per_bit, color=colors, width=0.8)
            ax.axhline(mean_h, color='#f9e2af', linewidth=1, linestyle='--', label=f'mean={mean_h:.3f}')
            ax.set_xlabel('Bit index', color='#cdd6f4', fontsize=9)
            ax.set_ylabel('Entropy', color='#cdd6f4', fontsize=9)
            ax.legend(fontsize=8, facecolor='#313244', labelcolor='#cdd6f4')
        ax.tick_params(colors='#6c7086', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#45475a')
        self._canvas.fig.tight_layout(pad=0.5)
        self._canvas.draw()

    # ── Language quality ───────────────────────────────────────────────

    def _display_language(self, metrics: dict, details: list):
        self._update_cards([
            ('Perplexity', f"{metrics.get('perplexity', 0):.2f}"),
            ('Samples', str(len(details))),
            ('—', '—'),
            ('—', '—'),
        ])

        headers = ['#', 'Generated Text']
        self._tbl_details.setColumnCount(len(headers))
        self._tbl_details.setHorizontalHeaderLabels(headers)
        self._tbl_details.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._tbl_details.setRowCount(0)
        for i, d in enumerate(details):
            row = self._tbl_details.rowCount()
            self._tbl_details.insertRow(row)
            self._tbl_details.setItem(row, 0, QTableWidgetItem(str(i + 1)))
            text = d if isinstance(d, str) else str(d)
            self._tbl_details.setItem(row, 1, QTableWidgetItem(text[:120]))

        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')
        ax.text(0.5, 0.5, f"PPL = {metrics.get('perplexity', 0):.2f}",
                ha='center', va='center', fontsize=28, color='#89b4fa',
                fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        self._canvas.draw()

    # ── Subsumption / generic ──────────────────────────────────────────

    def _display_subsumption(self, metrics: dict, details: list):
        self._display_generic(metrics, details)

    def _display_generic(self, metrics: dict, details: list):
        # Show up to 4 float metrics as cards
        float_metrics = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float))][:4]
        card_data = [(k, f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in float_metrics]
        while len(card_data) < 4:
            card_data.append(('—', '—'))
        self._update_cards(card_data)

        # Details as key-value table
        if details and isinstance(details[0], dict):
            cols = list(details[0].keys())[:6]
            self._tbl_details.setColumnCount(len(cols))
            self._tbl_details.setHorizontalHeaderLabels(cols)
            self._tbl_details.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self._tbl_details.setRowCount(0)
            for d in details:
                row = self._tbl_details.rowCount()
                self._tbl_details.insertRow(row)
                for c, col in enumerate(cols):
                    val = d.get(col, '')
                    self._tbl_details.setItem(row, c, QTableWidgetItem(
                        f"{val:.3f}" if isinstance(val, float) else str(val)[:60]
                    ))
        else:
            # Metrics as rows
            self._tbl_details.setColumnCount(2)
            self._tbl_details.setHorizontalHeaderLabels(['Metric', 'Value'])
            self._tbl_details.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self._tbl_details.setRowCount(0)
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    row = self._tbl_details.rowCount()
                    self._tbl_details.insertRow(row)
                    self._tbl_details.setItem(row, 0, QTableWidgetItem(str(k)))
                    self._tbl_details.setItem(row, 1, QTableWidgetItem(
                        f"{v:.4f}" if isinstance(v, float) else str(v)
                    ))

        # Bar chart of float metrics
        ax = self._canvas.ax
        ax.cla()
        ax.set_facecolor('#313244')
        if float_metrics:
            labels = [k for k, _ in float_metrics]
            vals = [v for _, v in float_metrics]
            colors = ['#89b4fa', '#a6e3a1', '#f9e2af', '#fab387'][:len(labels)]
            ax.bar(labels, vals, color=colors)
            ax.tick_params(colors='#cdd6f4', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#45475a')
        self._canvas.fig.tight_layout(pad=0.5)
        self._canvas.draw()

    # ------------------------------------------------------------------

    def _update_cards(self, data: list[tuple[str, str]]):
        """Update metric cards with (label, value) tuples."""
        for card, (label, value) in zip(self._cards, data):
            # Re-create label text via object name hack
            # The card label is fixed at init; update the value widget
            card.set_value(value)
            # Update the section label (first child)
            first_lbl = card.layout().itemAt(0).widget()
            if isinstance(first_lbl, QLabel):
                first_lbl.setText(label)
