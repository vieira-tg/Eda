"""Testes estatísticos formais: normalidade, homocedasticidade, comparação de grupos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.constants import ALPHA


@dataclass
class ResultadoTestes:
    """Resultado dos testes estatísticos consolidados."""

    itens: list[dict] = field(default_factory=list)  # cada item: nome/teste, p_valor, conclusão, status
    globais: dict[str, dict] = field(default_factory=dict)


def _testes_de_chi2_adefência_uniforme(df: pd.DataFrame) -> dict[str, dict]:
    """Teste Chi² de aderência à distribuição uniforme das classes."""
    res = {}
    for col in df.columns:
        serie = df[col].dropna()
        if serie.empty:
            continue
        if not isinstance(serie.dtype, pd.CategoricalDtype) and not pd.api.types.is_object_dtype(serie):
            continue
        # só faz sentido com baixa cardinalidade
        if serie.nunique() < 2 or serie.nunique() > 40:
            continue
        contagem = serie.value_counts()
        n_classes = int(contagem.count())
        if n_classes < 2:
            continue
        esperado = np.full(n_classes, contagem.sum() / n_classes)
        chi2, p = stats.chisquare(contagem.values, f_exp=esperado)
        res[col] = {
            "estatistica": float(chi2),
            "p_valor": float(p),
            "n_classes": n_classes,
            "uniforme (p≥0.05)": p >= ALPHA,
        }
    return res


def _teste_levene(df: pd.DataFrame, col_grupo: str, col_valor: str) -> dict:
    """Levene: homocedasticidade entre grupos."""
    try:
        dados = df[[col_grupo, col_valor]].dropna()
        grupos = dados[col_grupo].unique()
        grupos_valores = [dados.loc[dados[col_grupo] == g, col_valor].values for g in grupos]
        # se só houver um grupo, não dá para comparar
        if len(grupos) < 2:
            return {"nao_aplicavel": True}
        stat, p = stats.levene(*grupos_valores, center="median")
        return {"estatistica": float(stat), "p_valor": float(p), "variancias_iguais (p≥0.05)": p >= ALPHA}
    except Exception:
        return {"nao_aplicavel": True}


def _teste_anova_ou_kruskal(df: pd.DataFrame, col_grupo: str, col_valor: str) -> dict:
    """ANOVA (paramétrico) ou Kruskal-Wallis (não-paramétrico) unifator.
    Faz seleção pela magnitude da assimetria.
    """
    try:
        dados = df[[col_grupo, col_valor]].dropna()
        grupos = dados[col_grupo].unique()
        valores = [dados.loc[dados[col_grupo] == g, col_valor].values for g in grupos]
        if len(valores) < 2:
            return {"nao_aplicavel": True}
        # escolha: se todas as distribuições forem simétricas, ANOVA; senão Kruskal.
        skw = dados[col_valor].skew()
        if abs(skw) < 0.8:
            stat, p = stats.f_oneway(*valores)
            return {"teste": "ANOVA", "estatistica": float(stat), "p_valor": float(p), "grupos_iguais (p≥0.05)": p >= ALPHA}
        stat, p = stats.kruskal(*valores)
        return {"teste": "Kruskal-Wallis", "estatistica": float(stat), "p_valor": float(p), "grupos_iguais (p≥0.05)": p >= ALPHA}
    except Exception:
        return {"nao_aplicavel": True}


def avaliar_testes_estatisticos(df: pd.DataFrame, alvo: Optional[str] = None) -> ResultadoTestes:
    """Executa os principais testes e devolve o resultado consolidado.

    Colaterais:
        - Normalidade (por coluna categórica/não-categórica quando aplicável)
        - Homocedasticidade (por grupos)
        - Comparação de grupos (ANOVA/Kruskal) quando houver grupo
        - Chi² de uniformidade por coluna categórica
    """
    itens: list[dict] = []
    globais = {}

    # Chi² de aderência à uniforma em colunas categóricas
    globais["chi2_aderencia_uniforme"] = _testes_de_chi2_adefência_uniforme(df)

    # Se houver alvo categórico, aplicamos levene/anova em numéricos por grupo
    if alvo is not None and alvo in df.columns:
        if not pd.api.types.is_numeric_dtype(df[alvo]):
            # broadcasting do alvo por grupos
            for col in df.columns:
                if col == alvo:
                    continue
                if pd.api.types.is_numeric_dtype(df[col]):
                    result = _teste_levene(df, alvo, col)
                    if not result.get("nao_aplicavel"):
                        globais.setdefault("levene", {})[col] = result
                    result2 = _teste_anova_ou_kruskal(df, alvo, col)
                    if not result2.get("nao_aplicavel"):
                        globais.setdefault("anova_ou_kruskal", {})[col] = result2
        else:
            # alvo numérico: apenas normalidade por coluna
            # e podemos gerar displays de normalidade.
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                globais.setdefault("normalidade", {})[col] = {
                    "skewness": float(df[col].dropna().skew()),
                    "kurtosis": float(df[col].dropna().kurt()),
                }

    # itens (filhos simples para exibição rápida)
    for col, info in globais.get("chi2_aderencia_uniforme", {}).items():
        status = "ok" if info["uniforme (p≥0.05)"] else "warn"
        itens.append({
            "nome": f"Chi² uniformidade ({col})",
            "p_valor": info["p_valor"],
            "conclusao": "Uniforme" if info["uniforme (p≥0.05)"] else "Não-uniforme",
            "status": status,
        })
    for col, info in globais.get("levene", {}).items():
        status = "ok" if info["variancias_iguais (p≥0.05)"] else "warn"
        itens.append({
            "nome": f"Levene ({col} por alvo)",
            "p_valor": info["p_valor"],
            "conclusao": "Homocedástico" if info["variancias_iguais (p≥0.05)"] else "Heterocedástico",
            "status": status,
        })
    for col, info in globais.get("anova_ou_kruskal", {}).items():
        status = "ok" if info["grupos_iguais (p≥0.05)"] else "warn"
        itens.append({
            "nome": f"{info['teste']} ({col} por alvo)",
            "p_valor": info["p_valor"],
            "conclusao": f"Valores similares entre grupos ({info['teste']})" if info["grupos_iguais (p≥0.05)"] else f"Grupos diferentes ({info['teste']})",
            "status": status,
        })

    # normalidade (grupo false)
    if "normalidade" in globais:
        for col, info in globais["normalidade"].items():
            itens.append({
                "nome": f"Normalidade {col}",
                "p_valor": info.get("skewness", np.nan),
                "conclusao": f"Skew {info['skewness']:.3f}",
                "status": "ok",
            })

    return ResultadoTestes(itens=itens, globais=globais)
