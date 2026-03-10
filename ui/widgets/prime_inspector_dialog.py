"""
PrimeInspectorDialog — Click a prime, see all probe words that share it.
Editable probe vocabulary.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QTextEdit, QSplitter, QFrame, QListWidgetItem
)
from PySide6.QtCore import Qt


class PrimeInspectorDialog(QDialog):
    """Shows words sharing a prime factor + lets user edit probe vocabulary."""

    def __init__(self, iface, prime: int, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._prime = prime
        self.setWindowTitle(f"Prime Inspector -- p{prime}")
        self.resize(700, 500)
        self._setup_ui()
        self._refresh_words()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Header ──────────────────────────────────────
        hdr = QLabel(f"PRIME {self._prime} -- Words sharing this factor")
        hdr.setObjectName("sectionLabel")
        layout.addWidget(hdr)

        # ── Splitter: word list | vocabulary editor ─────
        splitter = QSplitter(Qt.Horizontal)

        # Left: words sharing this prime
        left = QFrame()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 8, 0)

        self._lbl_count = QLabel("0 words")
        self._lbl_count.setObjectName("statsLabel")
        left_l.addWidget(self._lbl_count)

        self._word_list = QListWidget()
        self._word_list.setAlternatingRowColors(True)
        left_l.addWidget(self._word_list)

        splitter.addWidget(left)

        # Right: editable probe vocabulary
        right = QFrame()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 0, 0, 0)

        lbl_vocab = QLabel("PROBE VOCABULARY (one word per line)")
        lbl_vocab.setObjectName("sectionLabel")
        right_l.addWidget(lbl_vocab)

        self._txt_vocab = QTextEdit()
        self._txt_vocab.setPlaceholderText("king\nqueen\ndog\ncat\n...")
        # Load current probe words
        current = self._iface.get_probe_words()
        self._txt_vocab.setPlainText('\n'.join(current))
        right_l.addWidget(self._txt_vocab)

        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply Vocabulary")
        btn_apply.clicked.connect(self._apply_vocab)
        btn_row.addWidget(btn_apply)

        self._lbl_vocab_status = QLabel(f"{len(current)} words")
        self._lbl_vocab_status.setObjectName("statsLabel")
        btn_row.addWidget(self._lbl_vocab_status)
        btn_row.addStretch()
        right_l.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([300, 400])
        layout.addWidget(splitter, stretch=1)

        # ── Close button ────────────────────────────────
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)

    def _refresh_words(self):
        """Show all probe words that have this prime active."""
        self._word_list.clear()
        words = self._iface.words_for_prime(self._prime)
        for w in words:
            self._word_list.addItem(w)
        self._lbl_count.setText(f"{len(words)} / {len(self._iface.get_probe_words())} words share prime {self._prime}")

    def _apply_vocab(self):
        """Parse text area and update probe vocabulary."""
        text = self._txt_vocab.toPlainText()
        words = [w.strip().lower() for w in text.split('\n') if w.strip()]
        # Deduplicate preserving order
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        self._lbl_vocab_status.setText(f"Encoding {len(unique)} words...")
        self._iface.set_probe_words(unique)
        self._lbl_vocab_status.setText(f"{len(unique)} words")
        self._refresh_words()
