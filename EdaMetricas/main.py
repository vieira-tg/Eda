"""Ponto de entrada do EdaMetricas."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Logging global do app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EdaMetricas")


def main() -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    # Sem folha de estilos customizada: o app segue o tema nativo do SO
    # (Windows, Linux, etc.) de onde está rodando.

    from ui.main_window import MainWindow

    janela = MainWindow()
    janela.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
