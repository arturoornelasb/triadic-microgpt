"""
MainWindow — QMainWindow with 7-tab interface and model panel header.
"""
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar, QLabel
)
from PySide6.QtCore import Qt

from ui.model_panel import ModelPanel
from ui.workers.model_worker import TaskWorker

# Lazy tab imports (avoid torch import before model is loaded)
def _import_tabs():
    from ui.tabs.encoder_tab import EncoderTab
    from ui.tabs.compare_tab import CompareTab
    from ui.tabs.explore_tab import ExploreTab
    from ui.tabs.analogy_tab import AnalogyTab
    from ui.tabs.validate_tab import ValidateTab
    from ui.tabs.chat_tab import ChatTab
    from ui.tabs.benchmarks_tab import BenchmarksTab
    return EncoderTab, CompareTab, ExploreTab, AnalogyTab, ValidateTab, ChatTab, BenchmarksTab


class _NoModelTab(QWidget):
    """Placeholder shown in tabs when no model is loaded."""
    def __init__(self, tab_name: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        lbl = QLabel(f"Load a model to use the «{tab_name}» tab")
        lbl.setObjectName("noModelLabel")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self._iface = None  # ModelInterface, set after loading
        self._tabs: list = []  # actual tab widgets
        self._warm_worker: TaskWorker | None = None
        self.setWindowTitle("TriadicGPT Explorer")
        self.resize(1280, 820)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Model panel (top bar) ──────────────────────────────
        self._model_panel = ModelPanel(self.project_root)
        self._model_panel.model_loaded.connect(self._on_model_loaded)
        self._model_panel.status_message.connect(self._set_status)
        layout.addWidget(self._model_panel)

        # ── Tab widget ─────────────────────────────────────────
        self._tabs_widget = QTabWidget()
        self._tabs_widget.setDocumentMode(True)

        tab_names = ['Encoder', 'Compare', 'Explore', 'Analogy', 'Validate', 'Chat', 'Benchmarks']
        for name in tab_names:
            placeholder = _NoModelTab(name)
            self._tabs_widget.addTab(placeholder, name)

        layout.addWidget(self._tabs_widget)

        # ── Status bar ─────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("No model loaded. Configure and load a model from the top bar.")

    # ------------------------------------------------------------------
    # Model loaded — replace placeholder tabs with real tabs
    # ------------------------------------------------------------------

    def _on_model_loaded(self, iface):
        self._iface = iface
        (EncoderTab, CompareTab, ExploreTab, AnalogyTab,
         ValidateTab, ChatTab, BenchmarksTab) = _import_tabs()

        real_tabs = [
            EncoderTab(iface),
            CompareTab(iface),
            ExploreTab(iface),
            AnalogyTab(iface),
            ValidateTab(iface),
            ChatTab(iface),
            BenchmarksTab(self.project_root),   # no model needed
        ]
        tab_names = ['Encoder', 'Compare', 'Explore', 'Analogy', 'Validate', 'Chat', 'Benchmarks']

        current = self._tabs_widget.currentIndex()
        for i, (tab, name) in enumerate(zip(real_tabs, tab_names)):
            self._tabs_widget.removeTab(0)
            self._tabs_widget.addTab(tab, name)

        self._tabs = real_tabs
        self._tabs_widget.setCurrentIndex(current)
        self._status_bar.showMessage(
            f"Model loaded: {iface.info_str} | {iface.device_str} | {iface.param_count/1e6:.1f}M params"
        )
        # Warm probe vocabulary in background
        self._warm_worker = TaskWorker(iface.warm_vocab)
        self._warm_worker.result_ready.connect(self._on_vocab_warmed)
        self._warm_worker.start()

    def _on_vocab_warmed(self, result):
        n = result.get('n_words', 0)
        n_probes = result.get('n_probes', 0)
        self._status_bar.showMessage(
            f"Model ready -- {n_probes} probe words indexed -- click any prime to inspect"
        )

    def _set_status(self, msg: str):
        self._status_bar.showMessage(msg)
