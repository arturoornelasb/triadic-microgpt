"""
BitVectorWidget — Visualizes a binary bit vector as colored squares.

64 squares in a grid:
  - Active bit (1):  green  #a6e3a1
  - Inactive bit (0): dark  #45475a
  - Hover shows tooltip: "Bit i — prime pᵢ"
  - Optional gradient mode using projection float values
"""
from PySide6.QtWidgets import QWidget, QToolTip
from PySide6.QtGui import QPainter, QColor, QFont, QPen
from PySide6.QtCore import Qt, QRect, QPoint


# First 128 primes (enough for k=128 bits)
_PRIMES_128 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
    137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373,
    379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
    461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557,
    563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
    643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719,
]

_COLOR_ACTIVE = QColor('#a6e3a1')       # green
_COLOR_INACTIVE = QColor('#313244')     # surface dark
_COLOR_BORDER = QColor('#45475a')       # overlay
_COLOR_HOVER = QColor('#89b4fa')        # blue accent
_COLOR_ACTIVE_SHARED = QColor('#a6e3a1')   # green (shared)
_COLOR_ACTIVE_ONLY_A = QColor('#f9e2af')   # yellow (only A)
_COLOR_ACTIVE_ONLY_B = QColor('#fab387')   # peach (only B)


class BitVectorWidget(QWidget):
    """Displays n_bits as colored squares."""

    def __init__(self, n_bits: int = 64, parent=None):
        super().__init__(parent)
        self.n_bits = n_bits
        self._bits: list[int] = [0] * n_bits
        self._projections: list[float] = [0.0] * n_bits
        self._mode = 'binary'   # 'binary' | 'gradient' | 'compare'
        self._compare_mask: list[str] = []  # 'shared'|'only_a'|'only_b'|'inactive' per bit
        self._hovered: int = -1
        self.setMouseTracking(True)
        self._cols = min(n_bits, 32)
        self._rows = (n_bits + self._cols - 1) // self._cols
        self._cell = 14  # px per cell
        self._gap = 2
        w = self._cols * (self._cell + self._gap) + self._gap
        h = self._rows * (self._cell + self._gap) + self._gap
        self.setFixedSize(w, h)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_bits(self, bits: list[int], projections: list[float] | None = None) -> None:
        """Update display with binary bits. Optional projections for gradient mode."""
        self._bits = list(bits)[:self.n_bits]
        if projections:
            self._projections = list(projections)[:self.n_bits]
            self._mode = 'gradient'
        else:
            self._mode = 'binary'
        self._compare_mask = []
        self.update()

    def set_compare(self, bits_a: list[int], bits_b: list[int]) -> None:
        """
        Show comparison between two bit vectors:
          - green  = both active (shared)
          - yellow = only A active
          - peach  = only B active
          - dark   = both inactive
        """
        mask = []
        for a, b in zip(bits_a, bits_b):
            if a and b:
                mask.append('shared')
            elif a and not b:
                mask.append('only_a')
            elif not a and b:
                mask.append('only_b')
            else:
                mask.append('inactive')
        self._compare_mask = mask
        self._bits = bits_a  # show A's bits
        self._mode = 'compare'
        self.update()

    def reset(self) -> None:
        self._bits = [0] * self.n_bits
        self._projections = [0.0] * self.n_bits
        self._compare_mask = []
        self._mode = 'binary'
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        g = self._gap
        c = self._cell

        for i, bit in enumerate(self._bits):
            col = i % self._cols
            row = i // self._cols
            x = g + col * (c + g)
            y = g + row * (c + g)
            rect = QRect(x, y, c, c)

            # Determine fill color
            if self._mode == 'compare' and self._compare_mask:
                status = self._compare_mask[i] if i < len(self._compare_mask) else 'inactive'
                if status == 'shared':
                    color = _COLOR_ACTIVE_SHARED
                elif status == 'only_a':
                    color = _COLOR_ACTIVE_ONLY_A
                elif status == 'only_b':
                    color = _COLOR_ACTIVE_ONLY_B
                else:
                    color = _COLOR_INACTIVE
            elif self._mode == 'gradient' and self._projections:
                val = self._projections[i] if i < len(self._projections) else 0.0
                if val > 0:
                    intensity = int(60 + val * 100)
                    color = QColor(max(0, 166 - intensity // 3), min(255, 227 - intensity // 4), max(0, 161 - intensity // 3))
                else:
                    color = _COLOR_INACTIVE
            else:
                color = _COLOR_ACTIVE if bit else _COLOR_INACTIVE

            if i == self._hovered:
                color = _COLOR_HOVER

            painter.setBrush(color)
            painter.setPen(QPen(_COLOR_BORDER, 1))
            painter.drawRoundedRect(rect, 2, 2)

        painter.end()

    def mouseMoveEvent(self, event):
        g = self._gap
        c = self._cell
        x, y = event.position().x(), event.position().y()
        col = int((x - g) // (c + g))
        row = int((y - g) // (c + g))
        idx = row * self._cols + col
        if 0 <= idx < self.n_bits:
            self._hovered = idx
            prime = _PRIMES_128[idx] if idx < len(_PRIMES_128) else '?'
            bit_val = self._bits[idx] if idx < len(self._bits) else 0
            proj_val = self._projections[idx] if idx < len(self._projections) else 0.0
            tip = f"Bit {idx} — prime p{idx}={prime} | val={proj_val:.3f} | {'active' if bit_val else 'inactive'}"
            QToolTip.showText(event.globalPosition().toPoint(), tip, self)
        else:
            self._hovered = -1
        self.update()

    def leaveEvent(self, event):
        self._hovered = -1
        self.update()
