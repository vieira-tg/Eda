"""Tabela de resumo para exibir estatísticas ou listas de itens com valores."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem, QWidget, QVBoxLayout,
)

from utils.tooltips import tooltip_metrica


class TabelaResumo(QWidget):
    """Wrapper simples de QTableWidget para exibir resumos por linha."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tabela = QTableWidget(self)
        self._tabela.setAlternatingRowColors(True)
        self._tabela.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tabela.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        # cabeçalho esticado: a tabela ocupar a largura toda, em vez de
        # aparecer espremida à esquerda com espaço vazio à direita
        cabecalho = self._tabela.horizontalHeader()
        cabecalho.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        cabecalho.setStretchLastSection(True)
        cabecalho.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._tabela.verticalHeader().setVisible(False)
        # mostrar o tooltip por mais tempo ao passar o mouse
        self._tabela.setToolTipDuration(15000)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabela)
        self.setLayout(layout)

    def limpar(self) -> None:
        self._tabela.setRowCount(0)
        self._tabela.setColumnCount(0)

    def set_linhas(
        self,
        cabecalho: list[str],
        linhas: list[dict[str, Any]],
        primeira_coluna_nome: str = "coluna",
    ) -> None:
        """Substitui o conteúdo por uma lista de dicts homogêneos.

        cabecalho: cabeçalho da tabela (o primeiro é o rótulo da linha).
        linhas: [{'coluna': X, '+outras colunas+': valor}, ...]
        primeira_coluna_nome: texto da primeira coluna (rótulo).
        """
        self.limpar()
        self._tabela.setColumnCount(len(cabecalho))
        # tooltips nos cabeçalhos: explicam cada métrica ao passar o mouse
        headers = []
        for cab in cabecalho:
            item = QTableWidgetItem(cab)
            dica = tooltip_metrica(cab)
            if dica:
                item.setToolTip(dica)
            headers.append(item)
        for c, item in enumerate(headers):
            self._tabela.setHorizontalHeaderItem(c, item)
        self._tabela.setRowCount(len(linhas))
        # a primeira coluna é o rótulo (key)
        for r, row in enumerate(linhas):
            for c, cab in enumerate(cabecalho):
                if c == 0:
                    texto = str(row.get(primeira_coluna_nome, ""))
                else:
                    texto = str(row.get(cab, ""))
                item = QTableWidgetItem(texto)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._tabela.setItem(r, c, item)

    def set_itens(self, itens: list[dict[str, Any]], cabecalho: Optional[list[str]] = None) -> None:
        """Modo alternativo — aceba uma lista de tuples/dicts e exiba."""
        if not itens:
            self.limpar()
            return
        # descobre as chaves do primeiro elemento
        if isinstance(itens[0], dict):
            keys = list(itens[0].keys())
            if cabecalho is None:
                cabecalho = keys
            self.set_linhas(cabecalho, itens, primeira_coluna_nome=keys[0])
        else:
            # fallback just show list
            dialogs = [[str(i), str(x)] for i, x in enumerate(itens)]
            self.set_linhas(["#", "valor"], [{"#": str(i), "valor": str(x)} for i, x in enumerate(itens)], primeira_coluna_nome="#")
