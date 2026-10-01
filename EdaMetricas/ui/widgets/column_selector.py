"""Painel lateral de seleção de colunas/features/target/tarefa."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget
)

from utils.tooltips import SELETOR


# Tarefas disponíveis — mapeamento rótulo visual -> chave interna
TAREFAS = {
    "Auto-detectar": "auto",
    "Regressão": "regressao",
    "Classificação": "classificacao",
    "Clusterização": "clusterizacao",
    "Série Temporal": "serie_temporal",
}


class ColumnSelector(QWidget):
    """
    - features: checkbox multi-seleção
    - alvo: combobox (uma coluna)
    - tempo: combobox opcional
    - tarefa: combobox

    Sinal ``solicitar_analise`` emite dict com o estado atual do painel.
    """

    solicitar_analise = Signal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checks: dict[str, QCheckBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Grupo de targets e tarefa
        grupo_cfg = QGroupBox("Definição", self)
        form = QFormLayout(grupo_cfg)

        self.combo_alvo = QComboBox(self)
        self.combo_alvo.addItem("<Nenhum>")
        self.combo_alvo.setToolTip(SELETOR["alvo"])
        form.addRow("Alvo (target):", self.combo_alvo)

        self.combo_tempo = QComboBox(self)
        self.combo_tempo.addItem("<Nenhum>")
        self.combo_tempo.setToolTip(SELETOR["tempo"])
        form.addRow("Coluna de tempo:", self.combo_tempo)

        self.combo_tarefa = QComboBox(self)
        for rotulo in TAREFAS.keys():
            self.combo_tarefa.addItem(rotulo)
        self.combo_tarefa.setToolTip(SELETOR["tarefa"])
        form.addRow("Tipo de tarefa:", self.combo_tarefa)

        layout.addWidget(grupo_cfg)

        # Grupo de features em scroll
        grupo_feats = QGroupBox("Features", self)
        grupo_feats.setToolTip(SELETOR["features"])
        scroll = QScrollArea(grupo_feats)
        scroll.setWidgetResizable(True)
        self._widget_feats = QWidget()
        self._layout_feats = QVBoxLayout(self._widget_feats)
        scroll.setWidget(self._widget_feats)
        grupo_feats.setLayout(QVBoxLayout())
        grupo_feats.layout().addWidget(scroll)
        layout.addWidget(grupo_feats, stretch=1)

        self.btn_analisar = QPushButton("Analisar", self)
        self.btn_analisar.clicked.connect(self._emitir_analise)
        layout.addWidget(self.btn_analisar)

    def _emitir_analise(self) -> None:
        self.solicitar_analise.emit(self.obter_selecao())

    # --------------------------------------------------------------
    def set_colunas(self, colunas: list[str]) -> None:
        """Reconstrói os checkboxes e combos a partir da lista de colunas."""
        # limpar checkboxes
        for ck in self._checks.values():
            ck.deleteLater()
        self._checks.clear()
        while self._layout_feats.count():
            item = self._layout_feats.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        # criar novos checkboxes
        for col in colunas:
            ck = QCheckBox(col, self._widget_feats)
            ck.setChecked(True)
            self._layout_feats.addWidget(ck)
            self._checks[col] = ck
        self._layout_feats.addStretch()

        # combos
        self.combo_alvo.clear()
        self.combo_alvo.addItem("<Nenhum>")
        self.combo_tempo.clear()
        self.combo_tempo.addItem("<Nenhum>")
        for col in colunas:
            self.combo_alvo.addItem(col)
            self.combo_tempo.addItem(col)

    def obter_selecao(self) -> dict:
        """Retorna o estado atual como dicionário da forma esperada pelos
        módulos de análise."""
        features = [c for c, ck in self._checks.items() if ck.isChecked()]
        alvo = self.combo_alvo.currentText()
        tempo = self.combo_tempo.currentText()
        tarefa_rotulo = self.combo_tarefa.currentText()
        return {
            "features": features,
            "alvo": alvo if not alvo.startswith("<") else None,
            "tempo_col": tempo if not tempo.startswith("<") else None,
            "tarefa": TAREFAS.get(tarefa_rotulo, "auto"),
        }
