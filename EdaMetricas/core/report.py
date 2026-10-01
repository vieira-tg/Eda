"""Geração de relatório HTML do EdaMetricas.

Produz um arquivo HTML autocontido (sem dependências externas), com:
- cabeçalho com data/hora e informações do dataset;
- tabelas de estatísticas, ausentes, outliers e testes (via pandas.to_html);
- gráficos embutidos em base64 (histograma do alvo, heatmap de correlação e
  barras de ausentes) gerados com matplotlib, sem GUI;
- o parecer final (veredito) das 4 tarefas.

Uso típico (sem Qt):

    gerar_relatorio(df, alvo, features, tempo_col, tarefa, caminho_saida)
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem GUI — necessário fora da janela Qt
from matplotlib.figure import Figure

from core.data_loader import DataLoader
from core.descriptive_stats import analisar_series
from core.missing_values import avaliar_ausentes
from core.correlation import analisar_correlacao
from core.outliers import analisar_outliers
from core.statistical_tests import avaliar_testes_estatisticos
from core.target_analysis import analisar_alvo
from core.temporal_analysis import analisar_temporal


# ---------------------------------------------------------------------------
# Utilitários de figura → base64
# ---------------------------------------------------------------------------

def _fig_para_base64(fig: Figure) -> str:
    """Renderiza uma figura matplotlib em PNG e devolve base64 para <img>."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fig_histograma(serie: pd.Series, titulo: str) -> Figure:
    fig = Figure(figsize=(6, 3.5), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.hist(serie.dropna(), bins=30, color="#1976D2", alpha=0.85)
    ax.set_title(titulo)
    return fig


def _fig_heatmap(matriz: pd.DataFrame, titulo: str) -> Figure:
    fig = Figure(figsize=(6.5, 5.5), tight_layout=True)
    ax = fig.add_subplot(111)
    im = ax.imshow(matriz.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(matriz.shape[1]))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(matriz.shape[0]))
    ax.set_yticklabels(matriz.index, fontsize=8)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


def _fig_barras_ausentes(pct_map: dict[str, float]) -> Figure:
    fig = Figure(figsize=(6.5, max(2.5, 0.35 * len(pct_map) + 1.5)), tight_layout=True)
    ax = fig.add_subplot(111)
    labels = list(pct_map.keys())
    ax.barh(range(len(labels)), [pct_map[c] for c in labels], color="#00ACC1")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% ausentes")
    ax.set_title("Percentual de ausentes por coluna")
    return fig


def _fig_barras_classes(contagens: dict, titulo: str) -> Figure:
    fig = Figure(figsize=(6, 3.5), tight_layout=True)
    ax = fig.add_subplot(111)
    labels = [k for k, v in contagens.items() if isinstance(v, dict)]
    vals = [contagens[k]["abs"] for k in labels]
    ax.bar(labels, vals, color="#1976D2")
    ax.set_title(titulo)
    ax.set_ylabel("contagem")
    return fig


# ---------------------------------------------------------------------------
# Geração do relatório
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #212121; }
h1 { color: #1976D2; } h2 { border-bottom: 2px solid #1976D2; padding-bottom: 4px; }
table { border-collapse: collapse; font-size: 13px; }
th { background: #1976D2; color: white; padding: 4px 8px; }
td { border: 1px solid #BDBDBD; padding: 3px 8px; }
tr:nth-child(even) { background: #F5F5F5; }
.ok { color: #388E3C; font-weight: bold; }
.warn { color: #FFA000; font-weight: bold; }
.error { color: #D32F2F; font-weight: bold; }
img { max-width: 700px; border: 1px solid #BDBDBD; margin: 8px 0; }
.meta { color: #616161; font-size: 12px; }
"""


def _selo_html(v) -> str:
    css = {"ok": "ok", "warn": "warn", "error": "error"}.get(v.status, "ok")
    return f'<span class="{css}">{v.selo} {v.status}</span>'


def gerar_relatorio_html(
    df: pd.DataFrame,
    caminho_origem: str,
    alvo: Optional[str] = None,
    features: Optional[list[str]] = None,
    tempo_col: Optional[str] = None,
    tarefa: Optional[str] = None,
) -> str:
    """Executa as análises e devolve o HTML do relatório como string.

    Esta função não depende de PySide6 e pode ser testada sem GUI.
    """
    loader = DataLoader()
    perfil = loader.perfil(df, caminho=caminho_origem)
    stats = analisar_series(df)
    ausentes = avaliar_ausentes(df)
    corr = analisar_correlacao(df, features=features, alvo=alvo)
    outl = analisar_outliers(df)
    alvo_res = analisar_alvo(df, alvo=alvo, forca_tarefa=tarefa)
    testes = avaliar_testes_estatisticos(df, alvo=alvo)
    from core.verdict_engine import veredito_geral
    veredito = veredito_geral(df, alvo=alvo, features=features, tempo_col=tempo_col, tarefa=tarefa)
    temporal = None
    serie_tempo = pd.to_datetime(df[tempo_col]) if (tempo_col and tempo_col in df.columns) else None
    if alvo and alvo in df.columns:
        temporal = analisar_temporal(df[alvo], serie_tempo=serie_tempo)
    elif serie_tempo is not None:
        num = list(df.select_dtypes("number").columns)
        if num:
            temporal = analisar_temporal(df[num[0]], serie_tempo=serie_tempo)

    partes: list[str] = []
    partes.append(f"<html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>")
    partes.append("<h1>EdaMetricas — Relatório de Análise Exploratória</h1>")
    partes.append(
        f"<p class='meta'>Gerado em {datetime.now():%d/%m/%Y %H:%M} | "
        f"Arquivo: {caminho_origem} | Tarefa: {tarefa or 'auto'}</p>"
    )

    # 1. Visão geral
    partes.append("<h2>1. Visão Geral</h2>")
    partes.append(f"""
    <table>
        <tr><td>Linhas</td><td>{perfil.linhas}</td></tr>
        <tr><td>Colunas</td><td>{perfil.colunas}</td></tr>
        <tr><td>Memória</td><td>{perfil.memoria_mb} MB</td></tr>
        <tr><td>Duplicatas</td><td>{perfil.n_duplicatas} ({perfil.percentual_duplicatas*100:.1f}%)</td></tr>
        <tr><td>Colunas numéricas</td><td>{', '.join(perfil.colunas_numericas) or '—'}</td></tr>
        <tr><td>Colunas categóricas</td><td>{', '.join(perfil.colunas_categoricas) or '—'}</td></tr>
        <tr><td>Colunas temporais</td><td>{', '.join(perfil.colunas_temporais) or '—'}</td></tr>
        <tr><td>Alta cardinalidade</td><td>{', '.join(perfil.colunas_id) or '—'}</td></tr>
        <tr><td>Constantes</td><td>{', '.join(perfil.colunas_constantes) or '—'}</td></tr>
    </table>""")

    # 2. Estatísticas descritivas — só numéricas em formato de tabela
    partes.append("<h2>2. Estatísticas Descritivas</h2>")
    linhas_num = []
    for col, info in stats.items():
        if info.get("tipo") == "numerica":
            linhas_num.append({
                "coluna": col, "média": info["media"], "mediana": info["mediana"],
                "std": info["std"], "mín": info["min"], "máx": info["max"],
                "IQR": info["iqr"], "skew": info["skewness"], "kurtose": info["kurtosis"],
            })
    if linhas_num:
        partes.append(pd.DataFrame(linhas_num).round(3).to_html(index=False))
    else:
        partes.append("<p>Sem colunas numéricas.</p>")

    # 3. Ausentes
    partes.append("<h2>3. Valores Ausentes</h2>")
    linhas_aus = [{
        "coluna": i["coluna"], "% ausentes": i["pct_ausentes"],
        "remover": "sim" if i.get("sugestao_remover") else "não",
        "estratégia": i.get("estrategia_imputacao", ""),
    } for i in ausentes.linhas]
    partes.append(pd.DataFrame(linhas_aus).to_html(index=False))
    partes.append(f"<img src='data:image/png;base64,{_fig_para_base64(_fig_barras_ausentes(ausentes.percentual_por_coluna))}'/>")

    # 4. Alvo
    partes.append("<h2>4. Variável Alvo</h2>")
    if alvo_res and alvo_res.alvo:
        partes.append(f"<p><b>Alvo:</b> {alvo_res.alvo} — tipo detectado: <b>{alvo_res.tipo}</b>.</p>")
        if alvo_res.tipo == "regressao":
            partes.append(f"<p>Skewness: {alvo_res.skewness:.3f}. Teste {alvo_res.testes_normalidade.get('nome','-')}: "
                          f"p={alvo_res.testes_normalidade.get('p_valor', float('nan')):.4f}</p>")
            partes.append("<ul>" + "".join(f"<li>{r}</li>" for r in alvo_res.recomendacao) + "</ul>")
            partes.append(f"<img src='data:image/png;base64,{_fig_para_base64(_fig_histograma(df[alvo], f'Distribuição de {alvo}'))}'/>")
        else:
            if alvo_res.contagens:
                partes.append(f"<img src='data:image/png;base64,{_fig_para_base64(_fig_barras_classes(alvo_res.contagens, 'Distribuição por classe'))}'/>")
            partes.append("<ul>" + "".join(f"<li>{r}</li>" for r in alvo_res.recomendacao) + "</ul>")
    else:
        partes.append("<p>Alvo não definido.</p>")

    # 5. Correlação
    partes.append("<h2>5. Correlação e Multicolinearidade</h2>")
    pearson = corr.get("pearson")
    if pearson is not None and not pearson.empty:
        partes.append(f"<img src='data:image/png;base64,{_fig_para_base64(_fig_heatmap(pearson, 'Correlação de Pearson'))}'/>")
    if corr.get("vif"):
        linhas_vif = [{"coluna": c, "VIF": v, "status": corr["vif_status"].get(c, "-")} for c, v in corr["vif"].items()]
        partes.append(pd.DataFrame(linhas_vif).to_html(index=False))
    ranking = corr.get("ranking_vs_alvo", [])
    if ranking:
        partes.append("<p><b>Ranking de features vs alvo:</b></p>")
        partes.append(pd.DataFrame(ranking).round(4).to_html(index=False))

    # 6. Outliers
    partes.append("<h2>6. Outliers</h2>")
    linhas_o = []
    for c, v in outl.get("por_coluna", {}).items():
        linhas_o.append({
            "coluna": c,
            "IQR": f"{v['iqr']['total']} ({v['iqr']['pct']}%)",
            "Z-score": f"{v['z_score']['total']} ({v['z_score']['pct']}%)",
        })
    partes.append(pd.DataFrame(linhas_o).to_html(index=False) if linhas_o else "<p>Sem colunas numéricas.</p>")

    # 7. Temporal
    partes.append("<h2>7. Análise Temporal</h2>")
    if temporal and temporal.n_validos >= 4:
        info = temporal.info
        est = "estacionária" if info.serie_estacionaria else "não estacionária"
        partes.append(
            f"<p>ADF: stat={info.adf_estatistica:.3f}, p={info.adf_p_valor:.4f} → <b>{est}</b>. "
            f"Frequência: {temporal.frequencia or '—'} | Período estimado: {info.periodo_estima or '—'} | "
            f"Ciclos: {f'{info.num_ciclos:.1f}' if info.num_ciclos else '—'}</p>"
        )
    else:
        partes.append("<p>Sem análise temporal disponível.</p>")

    # 8. Testes estatísticos
    partes.append("<h2>8. Testes Estatísticos</h2>")
    if testes.itens:
        partes.append(pd.DataFrame(testes.itens).to_html(index=False))
    else:
        partes.append("<p>Sem testes aplicáveis ao contexto atual.</p>")

    # 9. Veredito final
    partes.append("<h2>9. Veredito</h2>")
    for nome, obj in (("Regressão", veredito.reg), ("Classificação", veredito.cls),
                      ("Série Temporal", veredito.ser), ("Clusterização", veredito.clu)):
        partes.append(f"<p><b>{nome}:</b> {_selo_html(obj)}</p><ul>")
        partes.append("".join(f"<li>{r}</li>" for r in obj.razoes))
        partes.append("</ul>")

    partes.append("</body></html>")
    return "".join(partes)


def gerar_relatorio(
    df: pd.DataFrame,
    caminho_origem: str,
    caminho_saida: str | Path,
    alvo: Optional[str] = None,
    features: Optional[list[str]] = None,
    tempo_col: Optional[str] = None,
    tarefa: Optional[str] = None,
) -> Path:
    """Gera o relatório HTML e grava em disco. Retorna o caminho de saída."""
    html = gerar_relatorio_html(
        df, caminho_origem=caminho_origem, alvo=alvo,
        features=features, tempo_col=tempo_col, tarefa=tarefa,
    )
    caminho_saida = Path(caminho_saida)
    caminho_saida.write_text(html, encoding="utf-8")
    return caminho_saida
