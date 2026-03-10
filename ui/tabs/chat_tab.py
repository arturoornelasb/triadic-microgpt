"""
ChatTab — Tab 6: Conversation with real-time triadic verification panel.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QSplitter, QFrame, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QColor

from ui.widgets.bit_vector_widget import BitVectorWidget
from ui.workers.model_worker import TaskWorker


class ChatTab(QWidget):
    """
    Layout (horizontal splitter):
      LEFT (70%): chat history + input row
      RIGHT (30%): triadic analysis panel (last turn)
    """

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self._iface = iface
        self._worker: TaskWorker | None = None
        self._history: list[dict] = []  # [{"role": "user"|"ai", "text": str}]
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: Chat ────────────────────────────────────
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(12, 12, 6, 12)
        left_l.setSpacing(8)

        lbl_title = QLabel("CHAT")
        lbl_title.setObjectName("sectionLabel")
        left_l.addWidget(lbl_title)

        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setObjectName("chatDisplay")
        left_l.addWidget(self._chat_display, stretch=1)

        # Input row
        input_row = QHBoxLayout()
        self._txt_input = QLineEdit()
        self._txt_input.setPlaceholderText("Escribe tu mensaje...")
        self._txt_input.returnPressed.connect(self._send)
        input_row.addWidget(self._txt_input)

        self._btn_send = QPushButton("Enviar")
        self._btn_send.setObjectName("sendButton")
        self._btn_send.setMinimumWidth(80)
        self._btn_send.clicked.connect(self._send)
        input_row.addWidget(self._btn_send)

        btn_clear = QPushButton("Limpiar")
        btn_clear.setObjectName("smallButton")
        btn_clear.clicked.connect(self._clear_chat)
        input_row.addWidget(btn_clear)
        left_l.addLayout(input_row)

        self._lbl_status = QLabel("")
        self._lbl_status.setObjectName("statsLabel")
        left_l.addWidget(self._lbl_status)

        splitter.addWidget(left)

        # ── RIGHT: Triadic analysis panel ─────────────────
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6, 12, 12, 12)
        right_l.setSpacing(8)

        lbl_panel = QLabel("ANÁLISIS TRIÁDICO (último turno)")
        lbl_panel.setObjectName("sectionLabel")
        right_l.addWidget(lbl_panel)

        # Scroll area for the panel content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel_widget = QWidget()
        panel_l = QVBoxLayout(panel_widget)
        panel_l.setContentsMargins(4, 4, 4, 4)
        panel_l.setSpacing(8)

        # Prompt analysis
        lbl_prompt = QLabel("PREGUNTA")
        lbl_prompt.setObjectName("sectionLabel")
        panel_l.addWidget(lbl_prompt)

        self._lbl_prompt_phi = QLabel("Φ: —")
        self._lbl_prompt_phi.setObjectName("primeFactors")
        self._lbl_prompt_phi.setWordWrap(True)
        panel_l.addWidget(self._lbl_prompt_phi)

        self._bv_prompt = BitVectorWidget(n_bits=self._iface.n_bits)
        panel_l.addWidget(self._bv_prompt)

        # Response analysis
        lbl_resp = QLabel("RESPUESTA")
        lbl_resp.setObjectName("sectionLabel")
        panel_l.addWidget(lbl_resp)

        self._lbl_resp_phi = QLabel("Φ: —")
        self._lbl_resp_phi.setObjectName("primeFactors")
        self._lbl_resp_phi.setWordWrap(True)
        panel_l.addWidget(self._lbl_resp_phi)

        self._bv_resp = BitVectorWidget(n_bits=self._iface.n_bits)
        panel_l.addWidget(self._bv_resp)

        # Similarity
        sim_frame = QFrame()
        sim_frame.setObjectName("card")
        sim_l = QVBoxLayout(sim_frame)
        sim_l.setContentsMargins(8, 6, 8, 6)
        self._lbl_sim = QLabel("SIMILITUD: —")
        self._lbl_sim.setObjectName("statusLabel")
        sim_l.addWidget(self._lbl_sim)
        self._lbl_subsumes = QLabel("RESP ⊇ PREG: —")
        self._lbl_subsumes.setObjectName("statsLabel")
        sim_l.addWidget(self._lbl_subsumes)
        panel_l.addWidget(sim_frame)

        # Shared + gap
        lbl_shared = QLabel("COMPARTIDOS:")
        lbl_shared.setObjectName("sectionLabel")
        panel_l.addWidget(lbl_shared)
        self._lbl_shared_factors = QLabel("—")
        self._lbl_shared_factors.setObjectName("primeFactors")
        self._lbl_shared_factors.setWordWrap(True)
        panel_l.addWidget(self._lbl_shared_factors)

        lbl_gap = QLabel("RESP tiene de más:")
        lbl_gap.setObjectName("sectionLabel")
        panel_l.addWidget(lbl_gap)
        self._lbl_gap_factors = QLabel("—")
        self._lbl_gap_factors.setObjectName("primeFactors")
        self._lbl_gap_factors.setWordWrap(True)
        panel_l.addWidget(self._lbl_gap_factors)

        panel_l.addStretch()
        scroll.setWidget(panel_widget)
        right_l.addWidget(scroll)

        splitter.addWidget(right)
        splitter.setSizes([700, 300])
        layout.addWidget(splitter)

    # ------------------------------------------------------------------

    def _send(self):
        text = self._txt_input.text().strip()
        if not text:
            return
        self._txt_input.clear()
        self._btn_send.setEnabled(False)
        self._lbl_status.setText("Generando respuesta...")
        self._append_message("Usuario", text, '#89b4fa')

        self._worker = TaskWorker(self._iface.chat, text)
        self._worker.result_ready.connect(self._on_chat_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _append_message(self, role: str, text: str, color: str):
        cursor = self._chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._chat_display.setTextCursor(cursor)
        html = (
            f'<p><span style="color:{color}; font-weight:bold;">{role}:</span> '
            f'<span style="color:#cdd6f4;">{text.replace(chr(10), "<br>")}</span></p>'
        )
        self._chat_display.insertHtml(html)
        self._chat_display.ensureCursorVisible()

    def _clear_chat(self):
        self._chat_display.clear()
        self._history.clear()
        self._bv_prompt.reset()
        self._bv_resp.reset()
        self._lbl_prompt_phi.setText("Φ: —")
        self._lbl_resp_phi.setText("Φ: —")
        self._lbl_sim.setText("SIMILITUD: —")
        self._lbl_subsumes.setText("RESP ⊇ PREG: —")
        self._lbl_shared_factors.setText("—")
        self._lbl_gap_factors.setText("—")

    def _on_chat_result(self, result: dict):
        response = result.get('response', '')
        self._append_message("IA", response, '#a6e3a1')

        # Update triadic panel — interface returns flat dict
        p_phi = result.get('prompt_prime', 0)
        p_bits = result.get('prompt_bits', [])
        if p_phi:
            p_str = str(p_phi)
            if len(p_str) > 24:
                p_str = p_str[:12] + '...' + p_str[-6:]
            self._lbl_prompt_phi.setText(f"Φ: {p_str}")
        if p_bits:
            self._bv_prompt.set_bits(p_bits)

        # Response
        r_phi = result.get('response_prime', 0)
        r_bits = result.get('response_bits', [])
        if r_phi:
            r_str = str(r_phi)
            if len(r_str) > 24:
                r_str = r_str[:12] + '...' + r_str[-6:]
            self._lbl_resp_phi.setText(f"Φ: {r_str}")
        if r_bits:
            self._bv_resp.set_bits(r_bits)

        # Comparison
        sim = result.get('similarity', 0.0)
        sub = result.get('resp_subsumes_prompt', False)
        shared = result.get('shared_factors', [])
        gap = result.get('resp_extra_factors', [])

        self._lbl_sim.setText(f"SIMILITUD: {sim*100:.1f}%")
        self._lbl_subsumes.setText(f"RESP ⊇ PREG: {'✓ Sí' if sub else '✗ No'}")
        self._lbl_shared_factors.setText(
            ' '.join(str(p) for p in shared[:20]) if shared else '—'
        )
        self._lbl_gap_factors.setText(
            ' '.join(str(p) for p in gap[:20]) if gap else '—'
        )

        self._btn_send.setEnabled(True)
        self._lbl_status.setText("")

    def _on_error(self, msg: str):
        self._btn_send.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg.split(chr(10))[0]}")
        self._append_message("Error", msg.split('\n')[0], '#f38ba8')
