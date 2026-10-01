"""Janela principal do EdaMetricas."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QToolBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QStatusBar,
    QToolButton,
)

from core.data_loader import DataLoader, carregar_dataset
from core.descriptive_stats import analisar_series
from core.missing_values import avaliar_ausentes
from core.correlation import analisar_correlacao
from core.outliers import analisar_outliers
from core.sample_size import analisar_tamanho_amostral
from core.statistical_tests import avaliar_testes_estatisticos
from core.target_analysis import analisar_alvo
from core.temporal_analysis import analisar_temporal
from core.verdict_engine import veredito_geral
from utils.formatting import formatar_tempo
from utils.tooltips import BOTOES, ABAS as TOOLTIP_ABAS
from ui.widgets.chart_canvas import ChartCanvas
from ui.widgets.column_selector import ColumnSelector
from ui.widgets.summary_table import TabelaResumo

logger = logging.getLogger("EdaMetricas")


# ---------------------------------------------------------------------------
# Aba (visão simples)
# ---------------------------------------------------------------------------

class Aba(QWidget):
    """Aba padrão do app.

    Layout vertical compacto:
        [ título | (toolbar do matplotlib) ]  ← barra em uma linha
        [ canvas ]                            ← gráfico com altura limitada
        [ tabela ]                            ← fica logo abaixo, sempre visível
        [ texto de apoio ]
    """

    def __init__(self, titulo_aba: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        # ---- linha do título + toolbar do matplotlib ao lado direito ----
        linha_topo = QHBoxLayout()
        self.titulo = QLabel(titulo_aba, self)
        fonte = self.titulo.font()
        fonte.setBold(True)
        fonte.setPointSize(11)
        self.titulo.setFont(fonte)
        # toolbar do ChartCanvas criada aqui para ficar ao lado do título.
        # folha sem estilo custom: segue o tema nativo do SO
        self.canvas = ChartCanvas(self)
        tvbar = self.canvas.toolbar
        tvbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        linha_topo.addWidget(self.titulo)
        linha_topo.addStretch()
        linha_topo.addWidget(tvbar)
        layout.addLayout(linha_topo)

        # ---- gráfico com altura máxima moderada ----
        self.canvas.setMinimumHeight(260)
        self.canvas.setMaximumHeight(360)
        layout.addWidget(self.canvas)

        # ---- tabela logo em seguida; ocupa o restante ----
        self.tabela = TabelaResumo(self)
        self.tabela.setMinimumHeight(200)
        layout.addWidget(self.tabela, stretch=1)

        # ---- texto auxiliar pequeno no rodapé ----
        self.texto = QLabel("", self)
        self.texto.setWordWrap(True)
        layout.addWidget(self.texto)

    def limpar(self) -> None:
        self.canvas.limpar()
        self.tabela.limpar()
        self.texto.setText("")

    def set_tabela(self, cabecalho: list[str], linhas: list[dict], primeira_coluna_nome: str = "") -> None:
        if primeira_coluna_nome == "":
            primeira_coluna_nome = cabecalho[0]
        self.tabela.set_linhas(cabecalho, linhas, primeira_coluna_nome=primeira_coluna_nome)

    def set_texto(self, texto: str) -> None:
        self.texto.setText(texto)

    def limpar(self) -> None:
        self.canvas.limpar()
        self.tabela.limpar()
        self.texto.setText("")

    def set_tabela(self, cabecalho: list[str], linhas: list[dict], primeira_coluna_nome: str = "") -> None:
        if primeira_coluna_nome == "":
            primeira_coluna_nome = cabecalho[0]
        self.tabela.set_linhas(cabecalho, linhas, primeira_coluna_nome=primeira_coluna_nome)

    def set_texto(self, texto: str) -> None:
        self.texto.setText(texto)


class AbaVisaoGeral(Aba):
    """Resumo geral do dataset carregado (shape, memória, tipos etc.)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Visão Geral", parent)
        # bloco de texto logo abaixo do título
        self.resumo = QLabel("", self)
        self.resumo.setWordWrap(True)
        self.layout().insertWidget(1, self.resumo)

    def atualizar(self, perfil) -> None:
        linhas = [
            f"Shape: {perfil.linhas} linhas x {perfil.colunas} colunas",
            f"Memória: {perfil.memoria_mb:.3f} MB",
            f"Duplicatas: {perfil.n_duplicatas} ({perfil.percentual_duplicatas*100:.1f}%)",
            f"Colunas ID (alta cardinalidade): {', '.join(perfil.colunas_id) if perfil.colunas_id else '—'}",
            f"Colunas constantes: {', '.join(perfil.colunas_constantes) if perfil.colunas_constantes else '—'}",
            f"Colunas numéricas: {perfil.colunas_numericas or '—'}",
            f"Colunas categóricas: {perfil.colunas_categoricas or '—'}",
            f"Colunas temporais: {perfil.colunas_temporais or '—'}",
        ]
        self.resumo.setText("\n".join(linhas))


class AbaEstatisticas(Aba):
    def atualizar(self, stats: dict[str, dict]) -> None:
        linhas = []
        for col, info in stats.items():
            tipo = info.get("tipo", "desconhecido")
            linha = {"coluna": col, "tipo": tipo}
            if info.get("tipo") == "numerica":
                linha.update({
                    "media": info.get("media"),
                    "mediana": info.get("mediana"),
                    "std": info.get("std"),
                    "skewness": info.get("skewness"),
                    "kurtosis": info.get("kurtosis"),
                })
            elif info.get("tipo") == "categorica":
                linha.update({
                    "media": "",
                    "mediana": info.get("moda", ""),
                    "std": "",
                    "skewness": "",
                    "kurtosis": "",
                })
            else:
                linha.update({
                    "media": str(info.get("min", "")),
                    "mediana": str(info.get("max", "")),
                    "std": "",
                    "skewness": "",
                    "kurtosis": "",
                })
            linhas.append(linha)
        self.set_tabela(
            ["coluna", "tipo", "media", "mediana", "std", "skewness", "kurtosis"],
            linhas,
            primeira_coluna_nome="coluna",
        )


class AbaAusentes(Aba):
    def atualizar(self, resultado) -> None:
        linhas = []
        for info in resultado.linhas:
            linhas.append({
                "coluna": info["coluna"],
                "ausentes %": f"{info['pct_ausentes']:.1f}",
                "tipo": info.get("tipo", ""),
                "remover": "sim" if info.get("sugestao_remover") else "não",
                "estratégia": info.get("estrategia_imputacao", ""),
            })
        self.set_tabela(
            ["coluna", "ausentes %", "tipo", "remover", "estratégia"],
            linhas,
            primeira_coluna_nome="coluna",
        )
        try:
            dados = resultado.percentual_por_coluna
            labels = list(dados.keys())
            valores = [dados[c] for c in labels]
            self.canvas.plot_barras_h(labels, valores, titulo="Percentual de ausentes por coluna")
        except Exception:
            self.canvas.limpar()


class AbaAlvo(Aba):
    def atualizar(self, alvo) -> None:
        texto = []
        linhas = []
        if alvo.tipo == "regressao":
            texto.append(f"Tipo: REGRESSÃO (Skew={alvo.skewness:.3f})")
            texto.extend(alvo.recomendacao)
            teste = alvo.testes_normalidade or {}
            if teste:
                texto.append(f"Teste {teste.get('nome','?')}: p={teste.get('p_valor', float('nan')):.4f}")
        else:
            texto.append("Tipo: CLASSIFICAÇÃO")
            if alvo.contagens:
                for k, v in alvo.contagens.items():
                    if isinstance(v, dict):
                        linhas.append({
                            "classe": k,
                            "abs": v.get("abs", ""),
                            "pct": f"{v.get('pct', 0.0) * 100:.1f}%",
                        })
                texto.append(f"Classes: {len(linhas)}")
                texto.extend(alvo.recomendacao)
            if linhas:
                self.canvas.plot_barras(
                    [x["classe"] for x in linhas],
                    [float(x["pct"].rstrip('%')) for x in linhas],
                    titulo="Distribuição por classe",
                )
            else:
                self.canvas.limpar()
        self.tabela.set_linhas(["classe", "abs", "pct"], linhas, primeira_coluna_nome="classe")
        self.set_texto("\n".join(texto))


class AbaCorrelacao(Aba):
    def atualizar(self, resultado: dict) -> None:
        pearson = resultado.get("pearson")
        if pearson is not None and not pearson.empty:
            self.canvas.plot_heatmap(pearson, titulo="Correlação de Pearson")
        else:
            self.canvas.limpar()
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("Sem dados numéricos para correlação.")
            self.canvas.redesenhar()
        ranking = resultado.get("ranking_vs_alvo", [])
        linhas = [{"feature": x.get("feature") if isinstance(x, dict) else x} for x in ranking[:20]]
        self.tabela.set_linhas(["feature"], linhas, primeira_coluna_nome="feature")


class AbaOutliers(Aba):
    def atualizar(self, resultado: dict) -> None:
        por_coluna = resultado.get("por_coluna", {})
        linhas = []
        for c, v in por_coluna.items():
            iqr = v.get("iqr", {})
            z = v.get("z_score", {})
            linhas.append({
                "coluna": c,
                "IQR": f"{iqr.get('total', 0)} ({iqr.get('pct', 0.0):.1f}%)",
                "Z": f"{z.get('total', 0)} ({z.get('pct', 0.0):.1f}%)",
            })
        self.set_tabela(["coluna", "IQR", "Z"], linhas, primeira_coluna_nome="coluna")
        rs = resultado.get("resumo", {}) or {}
        if "isolation_forest" in rs:
            f = rs["isolation_forest"]
            self.set_texto(f"IsolationForest: {f.get('total',0)} outliers (~{f.get('pct',0.0):.1f}%)")
        else:
            self.set_texto(f"IQR: {rs.get('por_iqr',{})} | Z: {rs.get('por_zscore',{})}")


class AbaTemporal(Aba):
    def atualizar(self, res) -> None:
        # reproogie de dados em objeto de resultado da análise temporal
        texto = []
        info = getattr(res, "info", None)
        if info is not None:
            p = getattr(info, "adf_p_valor", float("nan"))
            if p == p:
                est_ac = "Estacionária" if p <= 0.05 else "Não estacionária"
                texto.append(f"ADF: p={p:.4f} → {est_ac}")
                if getattr(res, "frequencia", None):
                    texto.append(f"Frequência estimada: {res.frequencia}")
                if getattr(res, "periodo_estima", None):
                    texto.append(f"Período estimado: {res.periodo_estima}")
                if getattr(info, "num_ciclos", None):
                    texto.append(f"Ciclos estimados: {info.num_ciclos:.1f}")
        mono = getattr(res, "monotona", None) is not None and getattr(res, "monotona")
        if mono is not None:
            texto.append("Monotonia: " + ("Sim" if mono else "Não"))
        obs = getattr(res, "observacao", None)
        if obs:
            texto.append(f"Observação: {obs}")
        if num := getattr(res, "n_validos", None):
            texto.append(f"Observações válidas: {num}")
        self.set_texto("\n".join(texto))

        # desenho auxiliar (ACF/PACF) — apenas quando existir um InfoTemporal
        if info is not None and getattr(info, "acf", None):
            self.canvas.limpar()


class AbaVeredito(Aba):
    def atualizar(self, result) -> None:
        linhas = []
        for nome, objeto in (
            ("Regressão", result.reg),
            ("Classificação", result.cls),
            ("Série Temporal", result.ser),
            ("Clusterização", result.clu),
        ):
            linhas.append({
                "tarefa": nome,
                "status": objeto.status,
                "selo": objeto.selo,
                "razoes": " | ".join(objeto.razoes[:2]),
            })
        self.set_tabela(["tarefa", "status", "selo", "razoes"], linhas, primeira_coluna_nome="tarefa")


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Janela principal do aplicativo EdaMetricas."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EdaMetricas")
        self.resize(1280, 800)

        self.dataset: Optional[pd.DataFrame] = None
        self.perfil = None
        self._last_export_path: Optional[Path] = None

        self._build_ui()
        logger.info("EdaMetricas pronto")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout_principal = QVBoxLayout(central)

        # barra superior
        barra = QHBoxLayout()
        self.btn_abrir = QPushButton("Abrir Dataset", self)
        self.btn_abrir.setToolTip(BOTOES["abrir"])
        self.btn_abrir.clicked.connect(self._carregar)
        barra.addWidget(self.btn_abrir)

        self.combo_exemplo = QComboBox(self)
        self.combo_exemplo.setToolTip(BOTOES["combo_exemplo"])
        from core.sample_datasets import EXEMPLOS
        for nome in EXEMPLOS.keys():
            self.combo_exemplo.addItem(nome)
        lbl_exemplo = QLabel("Exemplo:", self)
        lbl_exemplo.setToolTip(BOTOES["combo_exemplo"])
        barra.addWidget(lbl_exemplo)
        barra.addWidget(self.combo_exemplo)

        self.btn_exemplo = QPushButton("Carregar Exemplo", self)
        self.btn_exemplo.setToolTip(BOTOES["carregar_exemplo"])
        self.btn_exemplo.clicked.connect(self._carregar_exemplo)
        barra.addWidget(self.btn_exemplo)

        self.btn_analisar = QPushButton("Analisar", self)
        self.btn_analisar.setToolTip(BOTOES["analisar"])
        self.btn_analisar.clicked.connect(self._analisar)
        self.btn_analisar.setEnabled(False)
        barra.addWidget(self.btn_analisar)

        self.btn_exportar = QPushButton("Exportar Relatório", self)
        self.btn_exportar.setToolTip(BOTOES["exportar"])
        self.btn_exportar.clicked.connect(self._exportar_relatorio)
        self.btn_exportar.setEnabled(False)
        barra.addWidget(self.btn_exportar)

        self.lbl_arquivo = QLabel("(nenhum arquivo carregado)", self)
        barra.addWidget(self.lbl_arquivo)
        barra.addStretch()
        layout_principal.addLayout(barra)

        # status bar
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)
        self.lbl_status = QLabel("Pronto", self)
        self.status.addPermanentWidget(self.lbl_status)

        # splitter para painel de seleção + área de abas
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout_principal.addWidget(splitter, stretch=1)

        self.selector = ColumnSelector(self)
        splitter.addWidget(self.selector)

        self.tabs = QTabWidget(self)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        # abas com títulos internos e áreas alinhadas
        self.aba_geral = AbaVisaoGeral()
        self.aba_estat = AbaEstatisticas("Estatísticas")
        self.aba_ausentes = AbaAusentes("Ausentes")
        self.aba_alvo = AbaAlvo("Alvo")
        self.aba_corr = AbaCorrelacao("Correlação")
        self.aba_outliers = AbaOutliers("Outliers")
        self.aba_temporal = AbaTemporal("Temporal")
        self.aba_veredito = AbaVeredito("Veredito")

        for titulo, widget in (
            ("Visão Geral", self.aba_geral),
            ("Estatísticas", self.aba_estat),
            ("Ausentes", self.aba_ausentes),
            ("Alvo", self.aba_alvo),
            ("Correlação", self.aba_corr),
            ("Outliers", self.aba_outliers),
            ("Temporal", self.aba_temporal),
            ("Veredito", self.aba_veredito),
        ):
            indice = self.tabs.addTab(widget, titulo)
            dica = TOOLTIP_ABAS.get(titulo)
            if dica:
                self.tabs.setTabToolTip(indice, dica)

    def showEvent(self, event) -> None:
        """Ao mostrar a janela, garante estado inicial coerente das abas."""
        super().showEvent(event)
        for aba in self._abas():
            aba.updateGeometry()

    def _abas(self) -> list:
        return [
            self.aba_geral, self.aba_estat, self.aba_ausentes, self.aba_alvo,
            self.aba_corr, self.aba_outliers, self.aba_temporal, self.aba_veredito,
        ]

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def _carregar(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self,
            "Abrir dataset",
            "",
            "Dados (*.csv *.xlsx *.xls *.parquet *.json);;Todos (*)",
        )
        if not caminho:
            return
        try:
            t0 = time.perf_counter()
            df, perfil = carregar_dataset(caminho)
            self.dataset = df
            self.perfil = perfil
            self.lbl_arquivo.setText(Path(caminho).name)
            self.selector.set_colunas(df.columns.tolist())
            self.btn_analisar.setEnabled(True)
            self.btn_exportar.setEnabled(True)
            self.aba_geral.atualizar(perfil)
            self.lbl_status.setText(f"Carregado: {Path(caminho).name} ({perfil.linhas} linhas)")
            logger.info("Dataset carregado: %s → %dx%d", caminho, perfil.linhas, perfil.colunas)
            # auto-seleciona alvo/tempo e preenche todas as abas
            self._auto_selecionar()
        except Exception as e:
            logger.exception("Erro ao carregar dataset")
            QMessageBox.critical(self, "Erro", f"Falha ao carregar {caminho}:\n{e}")

    def _auto_selecionar(self) -> None:
        """Escolhe automaticamente alvo e coluna de tempo e roda a análise.

        Heurística: alvo = coluna com 'alvo' no nome, senão a última numérica;
        tempo = primeira coluna detectada como temporal.
        """
        if self.dataset is None:
            return
        perfil = DataLoader().perfil(self.dataset)
        # tempo: primeira temporal nativa/inferida
        if perfil.colunas_temporais:
            self.selector.combo_tempo.setCurrentText(perfil.colunas_temporais[0])
        # alvo: prioriza nome com 'alvo', senão última numérica
        alvo = next(
            (c for c in perfil.colunas_numericas if "alvo" in c.lower()),
            (perfil.colunas_numericas[-1] if perfil.colunas_numericas else None),
        )
        if alvo:
            self.selector.combo_alvo.setCurrentText(alvo)
        self._analisar()

    def _carregar_exemplo(self) -> None:
        """Carrega um dos datasets sintéticos de exemplo (para uso rápido)."""
        from core.sample_datasets import EXEMPLOS

        nome = self.combo_exemplo.currentText()
        if nome not in EXEMPLOS:
            return
        try:
            df = EXEMPLOS[nome]()
            self.dataset = df
            self.perfil = DataLoader().perfil(df, caminho=f"<exemplo:{nome}>")
            self.lbl_arquivo.setText(f"<exemplo:{nome}>")
            self.selector.set_colunas(df.columns.tolist())
            self.btn_analisar.setEnabled(True)
            self.btn_exportar.setEnabled(True)
            self.aba_geral.atualizar(self.perfil)
            self.lbl_status.setText(f"Exemplo carregado: {nome} ({len(df)} linhas)")
            # seleciona alvo/tempo automaticamente e já preenche todas as abas
            self._auto_selecionar()
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível carregar o exemplo:\n{e}")

    def _analisar(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "Atenção", "Carregue um dataset antes de começar.")
            return

        sel = self.selector.obter_selecao()
        alvo = sel.get("alvo")
        tempo_col = sel.get("tempo_col")
        tarefa = sel.get("tarefa")
        features = [c for c in sel["features"] if c in self.dataset.columns]
        if not features:
            features = self.dataset.columns.tolist()
        cols = features.copy()
        if alvo and alvo not in cols:
            cols.append(alvo)
        if tempo_col and tempo_col not in cols:
            cols.append(tempo_col)
        df = self.dataset[cols].copy()

        t0 = time.perf_counter()
        try:
            self.aba_estat.atualizar(analisar_series(df))
            self.aba_ausentes.atualizar(avaliar_ausentes(df))
            self.aba_alvo.atualizar(analisar_alvo(df, alvo=alvo, forca_tarefa=tarefa))
            self.aba_corr.atualizar(analisar_correlacao(df, features=features, alvo=alvo))
            self.aba_outliers.atualizar(analisar_outliers(df))
            res_temp = analisar_temporal(df[alvo] if alvo in df else df.iloc[:, 0], serie_tempo=df[tempo_col] if tempo_col else None)
            self.aba_temporal.atualizar(res_temp)
            # veredicto com tamanho amostral (apenas se útil)
            try:
                _ = analisar_tamanho_amostral(df, alvo=alvo, tarefa=tarefa, tempo_col=tempo_col)
            except Exception:
                pass
            res = veredito_geral(df, alvo=alvo, features=features, tempo_col=tempo_col, tarefa=tarefa)
            self.aba_veredito.atualizar(res)
            self.btn_exportar.setEnabled(True)
            self._last_export_path = None
            self.lbl_status.setText(f"Análise concluída em {formatar_tempo(time.perf_counter() - t0)}")
        except Exception as e:
            logger.exception("Erro ao analisar dados")
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro durante a análise:\n{e}")

    def _exportar_relatorio(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "Atenção", "Carregue e analise um dataset primeiro.")
            return
        sel = self.selector.obter_selecao()
        alvo = sel.get("alvo")
        features = [c for c in sel["features"] if c in self.dataset.columns] if sel.get("features") else []
        from core.report import gerar_relatorio_html
        caminho_raw, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar relatório",
            "relatorio_eda.html",
            "Relatório HTML (*.html);;Todos os arquivos (*)",
        )
        if not caminho_raw:
            return
        try:
            html = gerar_relatorio_html(self.dataset, caminho_origem=str(self._dataset_name()), alvo=alvo,
                                       features=features, tempo_col=sel.get("tempo_col"), tarefa=sel.get("tarefa"))
            Path(caminho_raw).write_text(html, encoding="utf-8")
            QMessageBox.information(self, "Exportado", f"Relatório salvo em {caminho_raw}")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Falha ao exportar relatório:\n{e}")

    def _dataset_name(self) -> str:
        if self.perfil and getattr(self.perfil, "caminho", ""):
            return self.perfil.caminho
        return "<dataset>"
