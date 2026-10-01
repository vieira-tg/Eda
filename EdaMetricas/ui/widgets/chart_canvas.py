"""Canvas matplotlib embutido no PySide6 com toolbar de zoom/pan/salvar."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


# Paleta do app (alinha com utils.constants.PALETA)
PALETA = ["#1976D2", "#00ACC1", "#FFA000", "#D32F2F", "#388E3C"]


class ChartCanvas(QWidget):
    """Encapsula uma `matplotlib.figure.Figure` dentro de um widget Qt.

    Oferece métodos simples de plotagem reutilizados pelas abas da UI,
    todos com identidade visual consistente e renderização via Qt.
    """

    def __init__(self, parent: Optional[QWidget] = None, figsize: tuple = (7, 4.5)) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # a toolbar (zoom/pan/salvar) é criada aqui, mas NÃO entra no layout
        # interno — quem decide onde ela aparece é a aba (ao lado do título)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # começa oculto: só aparece quando um gráfico for desenhado
        self.setVisible(False)
        self.toolbar.setVisible(False)

    def limpar(self) -> None:
        """Limpa completamente a figura atual e oculta a área de gráfico."""
        self.figure.clear()
        self.canvas.draw_idle()
        self.setVisible(False)
        self.toolbar.setVisible(False)

    def redesenhar(self) -> None:
        """Atualiza o desenho e mostra a área somente se houver gráfico."""
        tem = self._tem_grafico()
        self.setVisible(tem)
        self.toolbar.setVisible(tem)
        self.canvas.draw_idle()

    def _tem_grafico(self) -> bool:
        """Só mostra o canvas/ferramentas se houver um gráfico desenhado."""
        return bool(self.figure.axes) and any(
            len(ax.lines) + len(ax.patches) + len(ax.collections) + len(ax.images) > 0
            for ax in self.figure.axes
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    def plot_histograma(self, serie: pd.Series, bins: int = 30, titulo: str = "") -> None:
        """Histograma simples com paleta padrão."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        dados = pd.Series(serie).dropna()
        ax.hist(dados, bins=bins, color=PALETA[0], alpha=0.8, edgecolor="none")
        if titulo:
            ax.set_title(titulo)
        self.redesenhar()

    def plot_barras(self, categorias: list[str], valores: list[float], titulo: str = "") -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(categorias, valores, color=PALETA[0])
        if titulo:
            ax.set_title(titulo)
        self.redesenhar()

    def plot_barras_h(self, rotulos: list[str], valores: list[float], titulo: str = "") -> None:
        """Barras horizontais — boa para ranking de colunas."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        pos = np.arange(len(rotulos))
        ax.barh(pos, valores, color=PALETA[1])
        ax.set_yticks(pos)
        ax.set_yticklabels(rotulos)
        ax.invert_yaxis()
        if titulo:
            ax.set_title(titulo)
        self.redesenhar()

    def plot_boxplot(self, serie: pd.Series, titulo: str = "") -> None:
        """Boxplot simples de uma série."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        dados = pd.Series(serie).dropna()
        ax.boxplot(dados, vert=False)
        if titulo:
            ax.set_title(titulo)
        self.redesenhar()

    def plot_heatmap(self, matriz: pd.DataFrame, titulo: str = "") -> None:
        """Heatmap de correlação com colorbar e rótulos curtos."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if matriz is None or matriz.empty:
            ax.set_title("Sem dados numéricos")
            self.redesenhar()
            return
        mat = matriz.values
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(matriz.shape[1]))
        ax.set_xticklabels(matriz.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(matriz.shape[0]))
        ax.set_yticklabels(matriz.index, fontsize=8)
        if titulo:
            ax.set_title(titulo)
        self.figure.colorbar(im, ax=ax, shrink=0.8)
        self.redesenhar()

    def plot_linha(self, x, series: dict[str, pd.Series], titulo: str = "") -> None:
        """Linha (potencialmente várias) por categoria/chave."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for i, (nome, y) in enumerate(series.items()):
            ax.plot(x, pd.Series(y).dropna().values, label=nome, color=PALETA[i % len(PALETA)])
        ax.legend()
        if titulo:
            ax.set_title(titulo)
        self.redesenhar()
