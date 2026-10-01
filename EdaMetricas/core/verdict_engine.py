"""Veredito automático: consolida as análises em um parecer final por tarefa.

Gera para cada uma das 4 tarefas (Regressão, Classificação, Clusterização,
Série Temporal) um selo (✅/⚠️/❌) e as razões do veredito.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.correlation import analisar_correlacao
from core.missing_values import avaliar_ausentes
from core.outliers import analisar_outliers
from core.target_analysis import analisar_alvo
from core.temporal_analysis import analisar_temporal
from utils.constants import (
    ALPHA,
    AUSENTES_CRITICO,
    AUSENTES_ALERTA,
    CLASS_MIN_CLASSE_ABSOLUTO,
    CLASS_MIN_POR_CLASSE_FEATURE,
    DESEQUILIBRIO_SEVEO,
    PERCENTUAL_ALTA_CARDINALIDADE,
    REGRA_REG_BASE,
    REGRA_REG_POR_FEATURE,
    VIF_ALERTA,
)

# rank: 0=ok, 1=warn, 2=error
RANK: dict[str, int] = {"ok": 0, "warn": 1, "error": 2}
ICONE = {"ok": "✅", "warn": "⚠️", "error": "❌"}


@dataclass
class ItemVeredicto:
    """Um item individual que entra no veredito final."""

    status: str = "ok"   # 'ok' | 'warn' | 'error'
    texto: str = ""


@dataclass
class VereditoTarefa:
    """Parecer de uma tarefa (Regressão/Classificação/Clusterização/Série)."""

    status: str = "ok"
    selo: str = "✅"
    razoes: list[str] = field(default_factory=list)


@dataclass
class VereditoGeral:
    """Resultado final, com pareceres para as 4 tarefas."""

    shape: str = ""
    alvo: str = ""
    tempo_col: str = ""
    tarefa: str = "auto"
    reg: VereditoTarefa = field(default_factory=VereditoTarefa)
    cls: VereditoTarefa = field(default_factory=VereditoTarefa)
    ser: VereditoTarefa = field(default_factory=VereditoTarefa)
    clu: VereditoTarefa = field(default_factory=VereditoTarefa)


# ---------------------------------------------------------------------------
# Regras de risco contra o enunciado (heurísticos simples).
# ---------------------------------------------------------------------------

def _msg_reg(regras: list[str]) -> VereditoTarefa:
    return VereditoTarefa(status="ok" if all(s == "ok" for s in []) else "warn", selo=ICONE["ok"], razoes=regras)


def _gera_melhor_acerto(items: list[ItemVeredicto]) -> VereditoTarefa:
    """Combina uma lista de ItemVeredicto no veredito da tarefa.
    A pior situação prevalece; a razão é a concatenação dos textos.
    """
    if not items:
        return VereditoTarefa(status="ok", selo=ICONE["ok"], razoes=["Sem barreiras conhecidas."])
    pior = max((RANK.get(x.status, 0) for x in items), default=0)
    selo = ICONE["ok" if pior == 0 else ("warn" if pior == 1 else "error")]
    status = "ok" if pior == 0 else ("warn" if pior == 1 else "error")
    razoes = [x.texto for x in items if x.texto]
    return VereditoTarefa(status=status, selo=selo, razoes=razoes)


def _status_item(status_ok: bool, texto_ok: str, texto_alarme: str) -> ItemVeredicto:
    if status_ok:
        return ItemVeredicto(status="ok", texto=texto_ok)
    return ItemVeredicto(status="warn", texto=texto_alarme)


# ---------------------------------------------------------------------------
# Vereditos específicos
# ---------------------------------------------------------------------------

def veredito_reg(df: pd.DataFrame, alvo: str | None, k_features: int, dados: dict) -> VereditoTarefa:
    """Veredito da tarefa de Regressão."""
    itens: list[ItemVeredicto] = []
    n = len(df)

    # 1. Amostragem mínima (Green)
    lim_geral = REGRA_REG_BASE + REGRA_REG_POR_FEATURE * k_features
    if n >= lim_geral:
        itens.append(ItemVeredicto("ok", f"Amostras satisfatórias: n={n} >= {lim_geral}."))
    elif n >= lim_geral * 0.5:
        itens.append(ItemVeredicto("warn", f"Amostras limitadas: n={n} vs {lim_geral}."))
    else:
        itens.append(ItemVeredicto("error", f"Amostra insuficiente: n={n} < {lim_geral}."))

    # 2. Ausentes / qualidade geral
    for info in dados.get("ausentes").por_coluna:
        if info["pct_ausentes"] >= AUSENTES_CRITICO:
            itens.append(ItemVeredicto("warn", f"{info['coluna']}: ausentes {info['pct_ausentes']:.1f}% — remover ou tratar."))
        elif info["pct_ausentes"] >= AUSENTES_ALERTA:
            itens.append(ItemVeredicto("warn", f"{info['coluna']}: ausentes {info['pct_ausentes']:.1f}% — imputar."))

    # 3. Correlações/multicolinearidade (VIF)
    for col, v in dados.get("correlacao", {}).get("vif_status", {}).items():
        if v == "error":
            itens.append(ItemVeredicto("warn", f"VIF crítico em {col}."))
        elif v == "warn":
            itens.append(ItemVeredicto("warn", f"VIF alto em {col}."))

    # 4. Distribuição do alvo
    if alvo is not None:
        split = dados.get("alvo")
        if split is not None and getattr(split, "tipo", "") == "regressao":
            if abs(split.skewness or 0) > 0.8:
                itens.append(ItemVeredicto("warn", "Alvo assimétrico (>|0.8|) — considerar transformação."))
            # normalidade: o paciente de testes normalidade
            t = split.testes_normalidade
            if not t.get("normal", True):
                itens.append(ItemVeredicto("warn", f"Teste '{t.get('nome','?')}' indica não normalidade."))
    else:
        itens.append(ItemVeredicto("error", "Alvo contínuo não definido."))

    return _gera_melhor_acerto(itens)


def veredito_cls(df: pd.DataFrame, alvo: str | None, k_features: int, dados: dict) -> VereditoTarefa:
    """Veredito da tarefa de Classificação."""
    itens: list[ItemVeredicto] = []
    n = len(df)

    # 1. Amostragem
    minimo = CLASS_MIN_POR_CLASSE_FEATURE * max(k_features, 1)
    if n >= minimo:
        itens.append(ItemVeredicto("ok", f"Amostras OK: n={n} ≥ {minimo} (recomendação mínima)."))
    else:
        itens.append(ItemVeredicto("warn", f"Amostras poucas: n={n} < {minimo}."))

    # 2. Ausentes
    for info in dados.get("ausentes").por_coluna:
        if info["pct_ausentes"] >= AUSENTES_CRITICO:
            itens.append(ItemVeredicto("warn", f"{info['coluna']}: ausentes {info['pct_ausentes']:.1f}% — remover ou tratar."))
        elif info["pct_ausentes"] >= AUSENTES_ALERTA:
            itens.append(ItemVeredicto("warn", f"{info['coluna']}: ausentes {info['pct_ausentes']:.1f}%-tratar."))

    # 3. Alvo
    alvo_t = dados.get("alvo")
    if alvo_t and getattr(alvo_t, "tipo", "") in ("classificacao", ""):
        # balanceamento
        contagens = alvo_t.contagens or {}
        if contagens:
            # apenas entradas de classe (dicts com 'abs'); ignora escalares como chi2
            cont_pair = [
                v["abs"] for v in contagens.values()
                if isinstance(v, dict) and "abs" in v
            ]
            if cont_pair:
                minimo_obs = min(cont_pair)
                ratio = max(cont_pair) / max(minimo_obs, 1)
                if ratio >= DESEQUILIBRIO_SEVEO:
                    itens.append(ItemVeredicto("warn", f"Desbalanceamento severo ({ratio:.1f}:1)."))
                elif ratio >= 3:
                    itens.append(ItemVeredicto("warn", f"Desbalanceamento leve ({ratio:.1f}:1)."))
    else:
        itens.append(ItemVeredicto("warn", "Alvo categórico abaixo de confirmação (preferência de configuração)."))

    # 4. Outliers — maior impacto em classificação
    for col, info in dados.get("outliers", {}).get("por_coluna", {}).items():
        if info["iqr"]["pct"] >= 5.0 or info["z_score"]["pct"] >= 5.0:
            itens.append(ItemVeredicto("warn", f"Outliers significativos em {col} (influencia rotações fronteira)."))
            break

    return _gera_melhor_acerto(itens)


def veredito_serie_temporal(df: pd.DataFrame, dados: dict) -> VereditoTarefa:
    """Veredito da tarefa de Série Temporal."""
    itens: list[ItemVeredicto] = []
    n = len(df)

    # 1. Amostras
    if n < 30:
        itens.append(ItemVeredicto("warn", "Série lenta (< 30) — análise frágil."))
    elif n < 60:
        itens.append(ItemVeredicto("ok", "Série razoável (30–60), un via efetiva."))
    else:
        itens.append(ItemVeredicto("ok", "Amostra suficiente para séries temporais."))

    # 2. ADF (coluna principal numérica ou a conhecida)
    adf = dados.get("adf", {})
    if adf.get("adf_p_valor") is not None and not np.isnan(adf.get("adf_p_valor")):
        if adf.get("adf_p_valor") <= ALPHA:
            itens.append(ItemVeredicto("ok", "Amostra estacionária (ADF p < 0.05)."))
        else:
            itens.append(ItemVeredicto("warn", "Amostra NÃO estacionária — considerar diferenciação/regressão com tendência."))

    # 3. Decomposição sazonal: requer 2-3 períodos
    periodo = dados.get("temporal", {}).get("periodo_estima")
    if periodo and periodo > 0:
        if n >= 2 * periodo:
            itens.append(ItemVeredicto("ok", f"Período estimado ~{periodo}; n ≥ 2·período."))
        else:
            itens.append(ItemVeredicto("warn", f"Período estimado ~{periodo}; n={n} insatisfatório (≥ 2·período ideal)."))

    return _gera_melhor_acerto(itens)


def veredito_clusterizacao(df: pd.DataFrame, dados: dict) -> VereditoTarefa:
    """Veredito da tarefa de Clusterização."""
    itens: list[ItemVeredicto] = []
    n = len(df)
    if n < 10:
        return VereditoTarefa("error", ICONE["error"], ["Amostra muito pequena para clusterização."])
    numericas = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    if n < 3 * numericas:
        itens.append(ItemVeredicto("warn", f"Maldição da dimensionalidade: n={n} < 3·k ({numericas})."))
    else:
        itens.append(ItemVeredicto("ok", f"Adequado: n={n}, features={numericas}."))
    # outlier density — pode impactar centroide
    dz = dados.get("outliers", {}).get("resumo", {}).get("iqr", {}).get("pct", 0)
    if dz >= 5.0:
        itens.append(ItemVeredicto("warn", f"Outliers representam ~{dz:.0f}% das amostras; agrupamento sensível."))
    return _gera_melhor_acerto(itens)


# ---------------------------------------------------------------------------
# Veredito geral reunindo os resultados enlatados.
# ---------------------------------------------------------------------------

def _veredito_com_resultados(
    df: pd.DataFrame,
    alvo: Optional[str],
    features: Optional[list[str]],
    tempo_col: Optional[str],
    tarefa: Optional[str],
) -> dict:
    """Executa todos os módulos 'core/' e retorna um dicionário de resultados."""
    numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if features:
        numericas = [c for c in features if c in numericas]
    k_features = len(numericas)

    # alvo é considerado, mas não contamos nas features se for numérica
    k_features_real = max(k_features - (1 if alvo is not None and alvo in numericas else 0), 0)

    dados: dict = {}
    dados["resumo"] = {
        "n": len(df),
        "k_features": k_features_real,
        "k_total": k_features,
    }

    dados["ausentes"] = avaliar_ausentes(df)

    if features is None:
        features = df.columns.tolist()

    # correlação considerando somente colunas numéricas e sem anas ao alvo fila
    dados["correlacao"] = analisar_correlacao(df, features=features, alvo=alvo)
    dados["outliers"] = analisar_outliers(df)
    dados["alvo"] = analisar_alvo(df, alvo=alvo, forca_tarefa=tarefa)

    # adf como resultado da coluna principal numérica
    if numericas:
        dados["adf"] = analisar_temporal(df[numericas[0]]).info.__dict__
    else:
        dados["adf"] = {}

    # temporal analysis: populate only if tempo_col present
    if tempo_col and tempo_col in df.columns:
        primeira = df[df[tempo_col].notna()].select_dtypes(include=[np.number]).copy()
        dados["temporal"] = analisar_temporal(primeira).info.__dict__ if not primeira.empty else {}
    else:
        dados["temporal"] = {}

    return dados


def veredito_geral(
    df: pd.DataFrame,
    alvo: Optional[str] = None,
    features: Optional[list[str]] = None,
    tempo_col: Optional[str] = None,
    tarefa: Optional[str] = None,
) -> VereditoGeral:
    """Retorna o veredito final com resultados consolidados em 4 tarefas.

    Os resultados calculados são salvos neste módulo para análise posterior.
    """
    dados = _veredito_com_resultados(df, alvo, features, tempo_col, tarefa)
    numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    k_features = len(numericas)
    if alvo is not None and alvo in numericas:
        k_features -= 1
    k_features = max(k_features, 0)

    reg = veredito_reg(df, alvo, k_features, dados)
    cls = veredito_cls(df, alvo, k_features, dados)
    ser = veredito_serie_temporal(df, dados)
    clu = veredito_clusterizacao(df, dados)

    return VereditoGeral(
        shape=f"{len(df)} x {df.shape[1]}",
        alvo=alvo or "",
        tempo_col=tempo_col or "",
        tarefa=tarefa or "auto",
        reg=reg,
        cls=cls,
        ser=ser,
        clu=clu,
    )
