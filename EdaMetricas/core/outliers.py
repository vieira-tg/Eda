"""Detecção de outliers: IQR, Z-score e Isolation Forest (opcional)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from utils.constants import OUTLIER_IQR_FATOR, OUTLIER_Z_LIMITE


@dataclass
class ResumoOutliers:
    """Resumo da detecção de outliers."""

    por_coluna: dict[str, dict] = field(default_factory=dict)
    isolamento: dict = field(default_factory=dict)  # resultado do IsolationForest


def _iqr_outliers(serie: pd.Series, fator: float = OUTLIER_IQR_FATOR) -> dict:
    """Outliers pelo método IQR (Tukey) e limites de acceptância."""
    serie = serie.dropna()
    if len(serie) < 4:
        return {"total": 0, "pct": 0.0, "limites": (np.nan, np.nan)}
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - fator * iqr
    lim_sup = q3 + fator * iqr
    mask = (serie < lim_inf) | (serie > lim_sup)
    total = int(mask.sum())
    return {
        "total": total,
        "pct": round(total / len(serie) * 100.0, 2),
        "limites": (float(lim_inf), float(lim_sup)),
    }


def _z_score_outliers(serie: pd.Series, limiar: float = OUTLIER_Z_LIMITE) -> dict:
    """Outliers pelo Z-score absoluto (> 3 por padrão)."""
    serie = serie.dropna()
    if len(serie) < 4:
        return {"total": 0, "pct": 0.0}
    z = np.abs(stats.zscore(serie, nan_policy="omit"))
    mask = z > limiar
    total = int(np.sum(mask))
    return {"total": total, "pct": round(total / len(serie) * 100.0, 2)}


def _isolation_forest(df: pd.DataFrame, colunas: list[str], contamination: float = 0.01) -> dict:
    """Detecção multivariada com Isolation Forest (rápido e não paramétrico)."""
    X = df[colunas].dropna()
    if X.shape[0] < 10 or X.shape[1] < 1:
        return {"total": 0, "pct": 0.0}
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(X.values)
        # -1 = outlier, 1 = inlier
        total = int((preds == -1).sum())
        return {"total": total, "pct": round(total / len(X) * 100.0, 2), "n_amostras": len(X)}
    except Exception:
        return {"total": 0, "pct": 0.0}


def analisar_outliers(
    df: pd.DataFrame,
    colunas: Optional[list[str]] = None,
    usar_isolation_forest: bool = True,
) -> dict:
    """Outliers em colunas numéricas. Retorna dict serializável."""
    if colunas is None:
        colunas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    resultado = {"por_coluna": {}, "resumo": {}}
    for col in colunas:
        serie = df[col].dropna()
        if serie.empty:
            resultado["por_coluna"][col] = {
                "iqr": {"total": 0, "pct": 0.0, "limites": (np.nan, np.nan)},
                "z_score": {"total": 0, "pct": 0.0},
                "tipo": "numerica",
            }
            continue
        resultado["por_coluna"][col] = {
            "iqr": _iqr_outliers(serie),
            "z_score": _z_score_outliers(serie),
            "tipo": "numerica",
        }

    # resumo global por método
    total_iqr = sum(x["iqr"]["total"] for x in resultado["por_coluna"].values())
    total_z = sum(x["z_score"]["total"] for x in resultado["por_coluna"].values())
    n_total = sum(len(df[c].dropna()) for c in colunas) or 1
    resultado["resumo"] = {
        "por_iqr": {"total": total_iqr, "pct": round(total_iqr / n_total * 100.0, 2)},
        "por_zscore": {"total": total_z, "pct": round(total_z / n_total * 100.0, 2)},
    }
    if usar_isolation_forest:
        num_cols = [c for c in colunas if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            resultado["resumo"]["isolation_forest"] = _isolation_forest(df, num_cols)
    return resultado
