"""Geração de datasets sintéticos de exemplo para testar o EdaMetricas.

Cada função devolve um DataFrame pequeno, realista e determinístico (seed
fixa), cobrindo os quatro cenários clássicos de uso do aplicativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exemplo_regressao(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Dataset de regressão: alvo contínuo com duas features informativas
    e uma feature irrelevante (ruído puro)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 3, n)          # informativa
    x2 = rng.normal(5, 2.5, n)         # informativa
    x3 = rng.normal(0, 1, n)           # irrelevante
    ruido = rng.normal(0, 2.5, n)
    preco = 3.0 * x1 + 1.5 * x2 + ruido
    return pd.DataFrame({
        "potencial_compra": x1,
        "renda_mensal": x2,
        "variancia_ruido": x3,
        "preco_arrouba": preco,
    })


def exemplo_classificacao(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Dataset de classificação binária com classes praticamente equilibradas."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y_freq = x + rng.normal(0, 0.6, n)
    classe = np.where(y_freq > 0.5, "alta", "baixa")
    return pd.DataFrame({
        "sinal_forca": x,
        "intensidade": rng.normal(0, 1, n) + 0.3 * x,
        "grupo": classe,
    })


def exemplo_serie_temporal(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Série temporal diária com tendência e sazonalidade semanal."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    serie = (
        0.05 * t                                  # tendência
        + 10.0 * np.sin(2 * np.pi * t / 7.0)      # sazonalidade semanal
        + rng.normal(0, 1.5, n)                   # ruído
    )
    return pd.DataFrame({
        "data": datas,
        "vendas_diarias": serie,
        "frota_rodando": rng.normal(120, 10, n),
    })


def exemplo_desbalanceado(n: int = 400, seed: int = 3, share_min: float = 0.10) -> pd.DataFrame:
    """Dataset de classificação com uma classe minoritária (~10%)."""
    rng = np.random.default_rng(seed)
    n_minor = int(n * share_min)
    n_major = n - n_minor
    f1 = np.concatenate([rng.normal(2, 1, n_major), rng.normal(-2, 1, n_minor)])
    f2 = rng.normal(0, 1, n)
    classe = np.array(["frequente"] * n_major + ["rara"] * n_minor)
    idx = rng.permutation(n)
    return pd.DataFrame({
        "medida_a": f1[idx],
        "medida_b": f2[idx],
        "classe": classe[idx],
    })


def exemplo_ausentes(n: int = 250, seed: int = 9) -> pd.DataFrame:
    """Dataset com ausências artificiais em algumas colunas."""
    rng = np.random.default_rng(seed)
    df = exemplo_regressao(n=n, seed=seed)
    df.loc[rng.choice(n, size=25, replace=False), "renda_mensal"] = np.nan    # ~10%
    df.loc[rng.choice(n, size=40, replace=False), "variancia_ruido"] = np.nan  # ~16%
    return df


def exemplo_completo(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Dataset que exercita TODAS as abas do app.

    Contém: coluna temporal diária, alvo contínuo (regressão), alvo binário
    (classificação), categórica, 3 numéricas (uma correlacionada → VIF alto),
    valores ausentes e outliers artificiais.
    """
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    t = np.arange(n)
    f1 = rng.normal(0, 1, n) + 0.5 * np.sin(2 * np.pi * t / 60.0)  # sazonalidade
    f2 = 2.0 * f1 + rng.normal(0, 0.1, n)                          # colinear com f1 (VIF alto)
    f3 = rng.normal(100, 20, n)
    ruido = rng.normal(0, 5, n)
    alvo_cont = 50 + 10 * f1 + 0.2 * f3 + ruido                    # alvo contínuo
    alvo_classe = np.where(alvo_cont > np.median(alvo_cont), "alto", "baixo")
    grupo = rng.choice(["A", "B", "C"], size=n)                    # categórica
    df = pd.DataFrame({
        "data": datas,
        "tendencia": f1,
        "colinear": f2,
        "volume_base": f3,
        "grupo": grupo,
        "alvo_regressao": alvo_cont,
        "alvo_classe": alvo_classe,
    })
    # injeta ausências (~10% em uma coluna, ~5% em outra)
    df.loc[rng.choice(n, size=n // 10, replace=False), "volume_base"] = np.nan
    df.loc[rng.choice(n, size=n // 20, replace=False), "colinear"] = np.nan
    # injeta outliers visíveis
    df.loc[rng.choice(n, size=8, replace=False), "alvo_regressao"] += 300
    return df


def exemplo_multicolinear(n: int = 300, seed: int = 11) -> pd.DataFrame:
    """Dataset com multicolinearidade forte para exibir alertas de VIF."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = x1 * 2 + rng.normal(0, 0.05, n)   # quase dependente de x1
    x3 = x1 + x2 + rng.normal(0, 0.05, n)  # combinação quase perfeita
    y = 0.5 * x1 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "resposta": y})


def exemplo_nao_estacionaria(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Random walk (não estacionária) em escala diária — ADF deve rejeitar H0."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2025-01-01", periods=n, freq="D")
    passe = np.cumsum(rng.normal(0, 1, n)) * 5 + 200
    return pd.DataFrame({"data": datas, "preco_dia": passe, "jornada": rng.normal(8, 1, n)})


# Mapeamento rótulo → função geradora usada pela UI
EXEMPLOS = {
    "Completo (todas as abas)": exemplo_completo,
    "Regressão": exemplo_regressao,
    "Classificação": exemplo_classificacao,
    "Série Temporal": exemplo_serie_temporal,
    "Desbalanceado": exemplo_desbalanceado,
    "Com ausências": exemplo_ausentes,
    "Multicolinear (VIF alto)": exemplo_multicolinear,
    "Não estacionária": exemplo_nao_estacionaria,
}


def carregar_exemplo(rotulo: str) -> pd.DataFrame:
    """Devolve o DataFrame sintético correspondente ao rótulo selecionado."""
    if rotulo not in EXEMPLOS:
        raise ValueError(f"Exemplo desconhecido: {rotulo}")
    return EXEMPLOS[rotulo]()
