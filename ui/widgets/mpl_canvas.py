"""
MplCanvas — Reusable matplotlib FigureCanvas embedded in Qt.
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):
    """Single-axes matplotlib canvas for embedding in PySide6 layouts."""

    def __init__(self, width: float = 5, height: float = 4, dpi: int = 100,
                 tight: bool = True, parent=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='#1e1e2e', edgecolor='none')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#313244')
        if tight:
            self.fig.tight_layout(pad=0.5)
        super().__init__(self.fig)
        self.setParent(parent)

    def clear(self):
        self.ax.cla()
        self.ax.set_facecolor('#313244')
        self.draw()
