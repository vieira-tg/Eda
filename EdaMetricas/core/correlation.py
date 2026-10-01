"""Correlação e multicolinearidade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils.constants import (
    ALPHA,
    COR_DEF_FORTE,
    VIF_ALERTA,
    VIF_CRITICO,
)


@dataclass
class ResumoCorrelacao:
    """Resumo da análise de correlação."""

    tabela: pd.DataFrame = field(default_factory=pd.DataFrame)
    vif: dict[str, float] = field(default_factory=dict)
    vif_status: dict[str, str] = field(default_factory=dict)  # 'ok'/'warn'/'error'
    valores_p: pd.DataFrame = field(default_factory=pd.DataFrame)


def _pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    return num.corr(method="pearson")


def _spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    return num.corr(method="spearman")


def _cramer_v(col1: pd.Series, col2: pd.Series) -> float:
    """Cramér's V para duas variáveis categóricas."""
    try:
        conf = pd.crosstab(col1, col2)
        chi2 = stats.chi2_contingency(conf)[0]
        n = conf.sum().sum()
        if n == 0:
            return float("nan")
        r, k = conf.shape
        phi2 = chi2 / n
        phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
        kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
        den = min((kcorr - 1), (rcorr - 1))
        return float(np.sqrt(phi2corr / den)) if den > 0 else float("nan")
    except Exception:
        return float("nan")


def _eta_squared(col_cat: pd.Series, col_num: pd.Series, dropna: bool = True) -> float:
    """Eta-squared (ANOVA de um fator) para associação categórica-numérica."""
    try:
        df = pd.DataFrame({"c": col_cat, "n": col_num}).dropna()
        if df.empty:
            return float("nan")
        # grupos
        grupos = df.groupby("c")["n"].mean().to_dict()
        overall = df["n"].mean()
        ss_between = sum(len(df[df["c"] == g]) * (m - overall) ** 2 for g, m in grupos.items())
        ss_total = ((df["n"] - overall) ** 2).sum()
        return float(ss_between / ss_total) if ss_total > 0 else float("nan")
    except Exception:
        return float("nan")


def _vif(X: pd.DataFrame) -> dict[str, float]:
    """VIF via statsmodels — requer dados sem ausências e com variância > 0."""
    try:
        X = X.dropna()
        if X.shape[0] < 10 or X.shape[1] < 2:
            return {}
        # ppv não-constante só contém float (mantém após remover constantes)
        cols = [c for c in X.columns if X[c].std() > 1e-9]
        if len(cols) < 2:
            return {}
        X = X[cols]
        # adicionar constante é necessário para o VIF, mas pode distorcer
        # por isso não adicionamos constante mantemos apenas explicativas.
        out: dict[str, float] = {}
        for i, col in enumerate(X.columns):
            try:
                out[col] = float(variance_inflation_factor(X.values, i))
            except Exception:
                out[col] = float("nan")
        return out
    except Exception:
        return {}


def _vif_status(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    if v >= VIF_CRITICO:
        return "error"
    if v >= VIF_ALERTA:
        return "warn"
    return "ok"


def analisar_correlacao(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    alvo: Optional[str] = None,
) -> dict:
    """Correlação e multicolinearidade em dados tabulares mistos.

    Retorna um dict com:
      - pearson/spearman/~cramers_v (par a par quando aplicável)
      - eta_squared (categórico-numérico)
      - VIF
      - ranking vs alvo quando o alvo for fornecido
    """
    df_local = df
    if features is not None:
        cols = [c for c in features if c in df_local.columns]
        if alvo is not None and alvo in df_local.columns:
            cols = list(dict.fromkeys(cols + [alvo]))
        df_local = df_local[cols]

    num = df_local.select_dtypes(include=[np.number])

    # matrizes
    pearson = num.corr(method="pearson") if not num.empty else pd.DataFrame()
    spearman = num.corr(method="spearman") if not num.empty else pd.DataFrame()

    # Cramér's V para categóricas (tolistas)
    cat_cols = [c for c in df_local.columns if not pd.api.types.is_numeric_dtype(df_local[c])]
    cramers_v: dict[tuple[str, str], float] = {}
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            ax, bx = cat_cols[i], cat_cols[j]
            cramers_v[(ax, bx)] = _cramer_v(df_local[ax], df_local[bx])

    # eta-squared par a par (categórico-numérico)
    numeric_cols = list(num.columns)
    numeric_only = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df_local[c])]
    eta_sq: dict[tuple[str, str], float] = {}
    for cat in cat_cols:
        for numcol in numeric_cols:
            eta_sq[(cat, numcol)] = _eta_squared(df_local[cat], df_local[numcol])

    # VIF apenas quando tem >= 2 numéricas e dimensão razoável
    vif = _vif(num) if num.shape[1] >= 2 else {}

    # Ranqueamento por |X| de correlação com o alvo (quando alvo fornecido)
    ranking_alto: list[dict] = []
    if alvo is not None and alvo in df_local.columns and not num.empty:
        # dependendo do tipo do alvo:
        if pd.api.types.is_numeric_dtype(df_local[alvo]):
            # alvo numérico -> correlação pearson/spearman com features numéricas
            for c in numeric_cols:
                if c == alvo:
                    continue
                pr = pearson.loc[c, alvo] if (c in pearson.index and alvo in pearson.columns) else np.nan
                sr = spearman.loc[c, alvo] if (c in spearman.index and alvo in spearman.columns) else np.nan
                ranking_alto.append({
                    "feature": c,
                    "pearson": float(pr) if not np.isnan(pr) else np.nan,
                    "spearman": float(sr) if not np.isnan(sr) else np.nan,
                })
        else:
            # alvo categórico: usar eta-squared e |V|
            for c in numeric_cols:
                eta = _eta_squared(df_local[alvo], df_local[c])
                v = _cramer_v(df_local[alvo].astype(str), df_local[c].astype(str))
                ranking_alto.append({
                    "feature": c,
                    "eta_squared": float(eta) if not np.isnan(eta) else np.nan,
                    "cramers_v": float(v) if not np.isnan(v) else np.nan,
                })
        # ordena por melhor mvítico disponível
        ranking_alto.sort(
            key=lambda x: max(
                abs(x.get("pearson", np.nan)) if not np.isnan(x.get("pearson", np.nan)) else 0.0,
                abs(x.get("spearman", np.nan)) if not np.isnan(x.get("spearman", np.nan)) else 0.0,
                abs(x.get("eta_squared", np.nan)) if not np.isnan(x.get("eta_squared", np.nan)) else 0.0,
                abs(x.get("cramers_v", np.nan)) if not np.isnan(x.get("cramers_v", np.nan)) else 0.0,
            ),
            reverse=True,
        )

    return {
        "pearson": pearson,
        "spearman": spearman,
        "cramers_v": cramers_v,
        "eta_squared": eta_sq,
        "vif": vif,
        "vif_status": {c: _vif_status(v) for c, v in vif.items()},
        "ranking_vs_alvo": ranking_alto,
        # resumos de texto úteis à interface
        "resumo": (
            f"Pearson/Spearman calculados entre {len(numeric_cols)} numéricas. "
            f"VIF calculado sobre {num.shape[1]} features."
        ),
    }
