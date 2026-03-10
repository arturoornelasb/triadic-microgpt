"""
ModelPanel — Top bar widget for loading models.

Supports:
  - Native TriadicGPT (checkpoint .pt + tokenizer.json)
  - HuggingFace model via TriadicWrapper (model name + optional weights)
  - Configurable n_bits, align_mode
"""
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QSpinBox, QProgressBar, QGroupBox, QFormLayout,
    QSizePolicy, QFrame
)
from PySide6.QtCore import Qt, Signal

from ui.workers.model_worker import ModelLoadWorker


class ModelPanel(QWidget):
    """
    Compact header bar for loading models.
    Emits model_loaded(ModelInterface) when a model is successfully loaded.
    """
    model_loaded = Signal(object)   # emits ModelInterface
    status_message = Signal(str)    # emits status text for status bar

    # Default paths relative to project root
    _DEFAULT_CKPT = 'checkpoints/torch_run15_strongalign/model_L12_D512_B64_best.pt'
    _DEFAULT_TOKN = 'checkpoints/torch_run15_strongalign/tokenizer.json'
    _DEFAULT_TRANSFER_CKPT = 'experiment10/checkpoints_infonce/phase_2_(unfreeze_last_layers)_final.pt'

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        self._worker: ModelLoadWorker | None = None
        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # ── Title row ──────────────────────────────────────────
        title_row = QHBoxLayout()
        lbl_title = QLabel("TriadicGPT Explorer")
        lbl_title.setObjectName("titleLabel")
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        self._lbl_status = QLabel("No model loaded")
        self._lbl_status.setObjectName("statusWarning")
        title_row.addWidget(self._lbl_status)
        outer.addLayout(title_row)

        # ── Config row ─────────────────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(8)

        # Backend selector
        row.addWidget(QLabel("Backend:"))
        self._cmb_backend = QComboBox()
        self._cmb_backend.addItem("TriadicGPT Native (.pt)", 'native')
        self._cmb_backend.addItem("GPT-2 Transfer (Exp10)", 'transfer')
        self._cmb_backend.addItem("HuggingFace (TriadicWrapper)", 'hf')
        self._cmb_backend.setFixedWidth(200)
        self._cmb_backend.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self._cmb_backend)

        # Checkpoint / HF model name
        self._lbl_ckpt = QLabel("Checkpoint:")
        row.addWidget(self._lbl_ckpt)
        self._txt_ckpt = QLineEdit()
        self._txt_ckpt.setPlaceholderText(str(self.project_root / self._DEFAULT_CKPT))
        self._txt_ckpt.setText(str(self.project_root / self._DEFAULT_CKPT))
        self._txt_ckpt.setFixedWidth(280)
        row.addWidget(self._txt_ckpt)
        btn_browse_ckpt = QPushButton("…")
        btn_browse_ckpt.setObjectName("smallButton")
        btn_browse_ckpt.setFixedWidth(28)
        btn_browse_ckpt.clicked.connect(self._browse_ckpt)
        row.addWidget(btn_browse_ckpt)

        # Tokenizer (native only)
        self._lbl_tok = QLabel("Tokenizer:")
        row.addWidget(self._lbl_tok)
        self._txt_tok = QLineEdit()
        self._txt_tok.setPlaceholderText(str(self.project_root / self._DEFAULT_TOKN))
        self._txt_tok.setText(str(self.project_root / self._DEFAULT_TOKN))
        self._txt_tok.setFixedWidth(200)
        row.addWidget(self._txt_tok)
        self._btn_browse_tok = QPushButton("…")
        self._btn_browse_tok.setObjectName("smallButton")
        self._btn_browse_tok.setFixedWidth(28)
        self._btn_browse_tok.clicked.connect(self._browse_tok)
        row.addWidget(self._btn_browse_tok)

        # n_bits
        row.addWidget(QLabel("Bits:"))
        self._spn_bits = QSpinBox()
        self._spn_bits.setRange(8, 128)
        self._spn_bits.setValue(64)
        self._spn_bits.setSingleStep(8)
        self._spn_bits.setFixedWidth(60)
        self._spn_bits.setToolTip("Number of triadic bits (HF backend only)")
        row.addWidget(self._spn_bits)

        # align_mode (HF only)
        self._lbl_align = QLabel("Align:")
        row.addWidget(self._lbl_align)
        self._cmb_align = QComboBox()
        self._cmb_align.addItems(['infonce', 'rank', 'mse'])
        self._cmb_align.setFixedWidth(90)
        self._cmb_align.setToolTip("Alignment loss mode (infonce=rich embeddings, mse=from-scratch)")
        row.addWidget(self._cmb_align)

        row.addStretch()

        # Load button
        self._btn_load = QPushButton("Load Model")
        self._btn_load.clicked.connect(self._load_model)
        self._btn_load.setMinimumWidth(120)
        row.addWidget(self._btn_load)

        outer.addLayout(row)

        # ── Progress bar ──────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        # Initial backend state
        self._on_backend_changed(0)

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _on_backend_changed(self, idx: int):
        backend = self._cmb_backend.itemData(idx)
        is_native = (backend == 'native')
        is_transfer = (backend == 'transfer')
        is_hf = (backend == 'hf')

        if is_native:
            self._lbl_ckpt.setText("Checkpoint:")
            self._txt_ckpt.setPlaceholderText(str(self.project_root / self._DEFAULT_CKPT))
            self._txt_ckpt.setText(str(self.project_root / self._DEFAULT_CKPT))
        elif is_transfer:
            self._lbl_ckpt.setText("Transfer .pt:")
            self._txt_ckpt.setPlaceholderText(str(self.project_root / self._DEFAULT_TRANSFER_CKPT))
            self._txt_ckpt.setText(str(self.project_root / self._DEFAULT_TRANSFER_CKPT))
        else:
            self._lbl_ckpt.setText("HF model or .pt:")
            self._txt_ckpt.setPlaceholderText("gpt2 / gpt2-medium / ...")

        # Show/hide tokenizer (only for native)
        self._lbl_tok.setVisible(is_native)
        self._txt_tok.setVisible(is_native)
        self._btn_browse_tok.setVisible(is_native)
        # HF-only controls
        self._lbl_align.setVisible(is_hf)
        self._cmb_align.setVisible(is_hf)
        self._spn_bits.setEnabled(not is_native)

    # ------------------------------------------------------------------
    # File browsers
    # ------------------------------------------------------------------

    def _browse_ckpt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", str(self.project_root / "checkpoints"),
            "PyTorch checkpoint (*.pt *.pth);;All (*)"
        )
        if path:
            self._txt_ckpt.setText(path)

    def _browse_tok(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tokenizer", str(self.project_root / "checkpoints"),
            "Tokenizer JSON (*.json);;All (*)"
        )
        if path:
            self._txt_tok.setText(path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        backend = self._cmb_backend.currentData()
        ckpt_path = self._txt_ckpt.text().strip()
        tok_path = self._txt_tok.text().strip() if backend == 'native' else ''
        n_bits = self._spn_bits.value()
        align_mode = self._cmb_align.currentText()

        # Validate paths for native and transfer backends
        if backend in ('native', 'transfer'):
            if not ckpt_path or not Path(ckpt_path).exists():
                self._set_status("Checkpoint not found", error=True)
                return
        if backend == 'native':
            if not tok_path or not Path(tok_path).exists():
                self._set_status("Tokenizer not found", error=True)
                return

        self._btn_load.setEnabled(False)
        self._progress.setVisible(True)
        self._set_status("Loading...", warning=True)

        if backend in ('native', 'transfer'):
            hf_name = ''
        else:
            hf_name = ckpt_path if not Path(ckpt_path).exists() else ''
            ckpt_path = ckpt_path if Path(ckpt_path).exists() else ''

        self._worker = ModelLoadWorker(
            backend=backend,
            checkpoint_path=ckpt_path,
            tokenizer_path=tok_path,
            n_bits=n_bits,
            align_mode=align_mode,
            hf_model_name=hf_name or ckpt_path,
        )
        self._worker.loaded.connect(self._on_loaded)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_loaded(self, iface):
        self._progress.setVisible(False)
        self._btn_load.setEnabled(True)
        info = iface.info_str
        device = iface.device_str
        params = f"{iface.param_count / 1e6:.1f}M params"
        self._set_status(f"✓ {info} | {device} | {params}", ok=True)
        self.status_message.emit(f"Model loaded: {info} | {device} | {params}")
        self.model_loaded.emit(iface)

    def _on_progress(self, msg: str):
        self._set_status(msg, warning=True)
        self.status_message.emit(msg)

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._btn_load.setEnabled(True)
        short = msg.split('\n')[0]
        self._set_status(f"❌ {short}", error=True)
        self.status_message.emit(f"Error: {short}")

    def _set_status(self, text: str, ok: bool = False, error: bool = False, warning: bool = False):
        self._lbl_status.setText(text)
        if ok:
            self._lbl_status.setObjectName("statusLabel")
        elif error:
            self._lbl_status.setObjectName("statusError")
        else:
            self._lbl_status.setObjectName("statusWarning")
        # Force style refresh
        self._lbl_status.style().unpolish(self._lbl_status)
        self._lbl_status.style().polish(self._lbl_status)
