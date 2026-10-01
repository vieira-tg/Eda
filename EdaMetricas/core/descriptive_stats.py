"""Estatísticas descritivas completas por coluna.

Comentários focados no 'porquê' de cada cálculo, para fins didáticos.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def _skew_pandas(serie: pd.Series) -> float:
    """Skewness (Fisher-Pearson, amostra corrigida)."""
    try:
        v = serie.skew()
        return float(v) if not pd.isna(v) else float("nan")
    except Exception:
        return float("nan")


def _kurtosis_pandas(serie: pd.Series) -> float:
    """Curtose (excesso) — pandas já devolve o *excess kurtosis*."""
    try:
        v = serie.kurt()
        return float(v) if not pd.isna(v) else float("nan")
    except Exception:
        return float("nan")


def _coeficiente_variacao(serie: pd.Series) -> float:
    """CV = desvio-padrão / média (em %), só faz sentido para dados razoáveis."""
    m = serie.mean()
    std = serie.std()
    if pd.isna(m) or pd.isna(std) or abs(m) < 1e-9:
        return float("nan")
    return float(std / m)


def analisar_coluna_numerica(serie: pd.Series) -> dict:
    """Estatísticas para uma coluna numérica.

    Retorna dicionário com: media, mediana, moda, std, var, min, max,
    amplitude, q1, q3, iqr, cv, skewness, kurtosis, n_ausentes e % ausentes.
    """
    serie = serie.dropna()
    n = len(serie)
    if n == 0:
        return {
            "media": np.nan, "mediana": np.nan, "moda": np.nan,
            "std": np.nan, "var": np.nan, "min": np.nan, "max": np.nan,
            "amplitude": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
            "cv": np.nan, "skewness": np.nan, "kurtosis": np.nan,
            "n_validos": 0, "ausentes": 0, "pct_ausentes": 0.0,
        }

    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    # mode robusto: pode haver mais de um; usamos o primeiro
    try:
        moda = serie.mode().iloc[0]
    except Exception:
        moda = np.nan

    return {
        "media": float(serie.mean()),
        "mediana": float(serie.median()),
        "moda": float(moda) if not pd.isna(moda) else float("nan"),
        "std": float(serie.std()),
        "var": float(serie.var()),
        "min": float(serie.min()),
        "max": float(serie.max()),
        "amplitude": float(serie.max() - serie.min()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "cv": _coeficiente_variacao(serie),
        "skewness": _skew_pandas(serie),
        "kurtosis": _kurtosis_pandas(serie),
        "n_validos": n,
        "ausentes": int(serie.isna().sum() + (len(serie) - n) * 0),  # placeholder
        "pct_ausentes": 0.0,  # placeholder
    }


def analisar_coluna_categorica(serie: pd.Series) -> dict:
    """Estatísticas para uma coluna categórica / discreta."""
    serie_nona = serie.dropna()
    n = len(serie_nona)
    freq = serie_nona.value_counts(dropna=True)
    rel = serie_nona.value_counts(normalize=True, dropna=True)
    top = freq.index[0] if len(freq) else None
    return {
        "n_validos": n,
        "ausentes": int(serie.isna().sum()),
        "pct_ausentes": float(serie.isna().mean()) * 100.0,
        "categorias_unicas": int(serie_nona.nunique()),
        "moda": top,
        # frequências como {valor: (abs, rel)}
        "frequencia": {str(k): (int(freq[k]), float(rel[k])) for k in freq.index},
    }


def analisar_coluna_datetime(serie: pd.Series) -> dict:
    """Estatísticas para uma coluna temporal (datetime)."""
    serie_nona = serie.dropna()
    if len(serie_nona) == 0:
        return {"n_validos": 0, "ausentes": int(serie.isna().sum())}
    return {
        "n_validos": len(serie_nona),
        "ausentes": int(serie.isna().sum()),
        "pct_ausentes": float(serie.isna().mean()) * 100.0,
        "min": serie_nona.min(),
        "max": serie_nona.max(),
        "amplitude": (serie_nona.max() - serie_nona.min()),
        "monotona": bool(serie_nona.sort_values().is_monotonic_increasing),
    }


def analisar_series(df: pd.DataFrame, coluna: Optional[str] = None) -> dict[str, dict]:
    """Analiza todas as colunas do DataFrame, ou apenas a coluna informada.

    Retorna um dicionário {coluna: {stats...}}.
    """
    if coluna is not None:
        serie = df[coluna]
        if pd.api.types.is_numeric_dtype(serie):
            return {coluna: {"tipo": "numerica", **analisar_coluna_numerica(serie)}}
        if pd.api.types.is_datetime64_any_dtype(serie):
            return {coluna: {"tipo": "temporal", **analisar_coluna_datetime(serie)}}
        return {coluna: {"tipo": "categorica", **analisar_coluna_categorica(serie)}}

    resultado: dict[str, dict] = {}
    for col in df.columns:
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            resultado[col] = {"tipo": "numerica", **analisar_coluna_numerica(serie)}
        elif pd.api.types.is_datetime64_any_dtype(serie):
            resultado[col] = {"tipo": "temporal", **analisar_coluna_datetime(serie)}
        else:
            resultado[col] = {"tipo": "categorica", **analisar_coluna_categorica(serie)}
    return resultado
