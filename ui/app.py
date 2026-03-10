"""
TriadicGPT Explorer — Desktop application entry point.

Usage:
    cd C:/Github/triadic-microgpt
    python ui/app.py
"""
import sys
import matplotlib
matplotlib.use('QtAgg')   # Must be set before any other matplotlib import

from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import MainWindow


def main():
    # High-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("TriadicGPT Explorer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("TriadicGPT")

    # Load stylesheet
    qss_path = Path(__file__).parent / 'resources' / 'style.qss'
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding='utf-8'))

    project_root = Path(__file__).parent.parent
    window = MainWindow(project_root)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
