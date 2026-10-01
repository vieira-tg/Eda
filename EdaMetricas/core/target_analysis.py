"""Análise da variável alvo: regressão vs classificação.

Detecta automaticamente se o alvo é contínuo (→ regressão) ou categórico/
discreto com baixa cardinalidade (→ classificação); permite override manual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.gofplots import qqplot

from utils.constants import (
    ALPHA,
    CLASS_MIN_CLASSE_ABSOLUTO,
    DESEQUILIBRIO_ALERTA,
    DESEQUILIBRIO_SEVEO,
    MAX_CLASSES_ALVO_DEFAULT,
    SKEWNESS_LIMITE_ALTO,
)


@dataclass
class ResultadoAlvo:
    """Resumo da natureza do alvo e associação com amostragens."""

    alvo: str = ""
    # 'regressao' | 'classificacao' | 'binaria' | 'categorica'
    tipo: str = ""
    # ação recomendada — transformações, alertas, etc.
    recomendacao: list[str] = field(default_factory=list)
    # detalhes ao usuário
    skewness: float = float("nan")
    testes_normalidade: dict = field(default_factory=dict)
    contagens: dict = field(default_factory=dict)


def _teste_normalidade(serie: pd.Series) -> dict[str, float]:
    """Testes de normalidade (escolha conforme tamanho).

    - n < 5000 → Shapiro-Wilk (o melhor p/ pequenas amostras),
    - n >= 5000 → Kolmogorov-Smirnov + Lilliefors implícito.
    """
    serie = serie.dropna()
    n = len(serie)
    if n < 5:
        return {"nome": "insuficiente", "p_valor": float("nan"), "normal": None}
    if n < 5000:
        stat, p = stats.shapiro(serie)
        return {"nome": "Shapiro-Wilk", "p_valor": float(p), "normal": p >= ALPHA}
    # D'Agostino K²
    stat, p = stats.normaltest(serie)
    return {"nome": "D'Agostino K²", "p_valor": float(p), "normal": p >= ALPHA}


def _recomendar_transformacao(serie: pd.Series, skew: float) -> list[str]:
    """Sugere transformações conforme a assimetria."""
    if np.isnan(skew):
        return []
    if abs(skew) < SKEWNESS_LIMITE_ALTO:
        return ["Sem transformação necessária."]
    rec = []
    rec.append("Distribuição assimétrica — considere transformação:")
    (caudas := [])  # guarda preenchimento
    if skew > SKEWNESS_LIMITE_ALTO:
        rec.append("- Yeo-Johnson/log1p se houver valores negativos.")
        rec.append("- log/Box-Cox se não houver valores negativos.")
    else:
        rec.append("- sqrt/Yeo-Johnson quando a cauda pesada é na esquerda.")
    rec.append("- Caso o alvo seja ordinal, muda-se para transformação/odd transformada.")
    return rec


def analisar_alvo(
    df: pd.DataFrame,
    alvo: Optional[str] = None,
    forca_tarefa: Optional[str] = None,
) -> ResultadoAlvo:
    """Detecta se o alvo é regresivo ou classificação e aplica as análises."""
    if alvo is None:
        return ResultadoAlvo(tipo="categorica", alvo=alvo or "", recomendacao=[])

    serie = df[alvo]
    serie_got = serie.dropna()
    n = len(serie_got)
    n_unique = serie_got.nunique()
    dtype_numeric = pd.api.types.is_numeric_dtype(serie)

    # heurística: alvo com baixa cardinalidade -> classificação.
    # inclui alvo categórico real. Caso ambíguo, preferimos claramente regressão.
    if forca_tarefa == "regressao":
        tipo = "classificacao" if (not dtype_numeric or n_unique < CLASS_MIN_CLASSE_ABSOLUTO) else "regressao"
    elif forca_tarefa == "classificacao":
        tipo = "regressao" if dtype_numeric and n_unique > CLASS_MIN_CLASSE_ABSOLUTO else "classificacao"
    else:
        # auto
        if dtype_numeric and n_unique > CLASS_MIN_CLASSE_ABSOLUTO:
            tipo = "regressao"
        else:
            tipo = "classificacao"

    if tipo == "binaria":
        pass  # reservado

    res = ResultadoAlvo(alvo=alvo, tipo=tipo)

    if tipo == "regressao":
        res.skewness = float(serie_got.skew()) if n >= 3 else float("nan")
        res.testes_normalidade = _teste_normalidade(serie_got)
        res.recomendacao = _recomendar_transformacao(serie_got, res.skewness)
        if res.testes_normalidade.get("normal") is False:
            res.recomendacao.append(
                "Teste de normalidade reprovado; ajustar com normalização/scaling após transformação."
            )
    else:
        # classificação
        contagens = serie_got.astype(str).value_counts()
        proporcao = serie_got.astype(str).value_counts(normalize=True)
        res.contagens = {
            str(k): {"abs": int(v), "pct": float(proporcao[k])} for k, v in contagens.items()
        }
        # desbalanceamento
        razao = float(contagens.max() / max(contagens.min(), 1))
        if razao >= DESEQUILIBRIO_SEVEO:
            res.recomendacao.append(
                f"Desbalanceamento severo ({razao:.1f}:1): considere SMOTE/class_weight/stratificação."
            )
        elif razao >= DESEQUILIBRIO_ALERTA:
            res.recomendacao.append(f"Desbalanceamento leve ({razao:.1f}:1).")
        # Chi-squared uniforme
        if n_classes := len(contagens):
            obs = contagens.values
            Esp = np.full(n_classes, obs.sum() / n_classes)
            chi2, p = stats.chisquare(obs, f_exp=Esp)
            res.contagens["chi2"] = float(chi2)
            res.contagens["chi2_p_valor"] = float(p)
    return res
