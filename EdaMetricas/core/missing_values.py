"""Avaliação de valores ausentes por coluna + sugestão de estratégia."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.constants import AUSENTES_ALERTA, AUSENTES_CRITICO


@dataclass
class VerdictAmbientes:
    """Resultado da avaliação de ausentes."""

    status: str = "ok"
    # tabela de colunas com {coluna, n_ausentes, pct_ausentes, tipo, sugestao}
    linhas: list[dict] = field(default_factory=list)
    # percentis legíveis (float) para o heatmap e semáforos
    percentual_por_coluna: dict[str, float] = field(default_factory=dict)
    # sugestao de estratégia por coluna
    por_coluna: list[dict] = field(default_factory=list)


def _sugerir_imputacao(tipo: str, pct: float) -> str:
    """Sugere estratégia de imputação por tipo de dado."""
    if pct == 0.0:
        return "Nenhuma"
    if tipo == "numerica":
        return "Média/mediana (na mediação geral) ou median em baixa dispersão."
    if tipo == "categorica":
        return "Moda (ou categorias provisória, ex.: 'desconhecido')."
    if tipo == "temporal":
        return "Forward/back fill ou interpolação temporal."
    return "Avaliar manualmente."


def avaliar_ausentes(df: pd.DataFrame) -> VerdictAmbientes:
    """Avalia % de ausentes por coluna e sinaliza candidatas a remoção."""
    n = len(df)
    if n == 0:
        return VerdictAmbientes(status="error", linhas=[])

    linhas: list[dict] = []
    pct_map: dict[str, float] = {}
    status_geral = "ok"

    for col in df.columns:
        n_aus = int(df[col].isna().sum())
        pct = float((n_aus / n) * 100.0)
        pct_map[col] = pct
        linhas.append({
            "coluna": col,
            "n_ausentes": n_aus,
            "pct_ausentes": round(pct, 2),
        })
        if pct >= AUSENTES_CRITICO:
            status_geral = "warn" if status_geral == "ok" else status_geral

    # ordena por % ausentes
    linhas.sort(key=lambda x: x["pct_ausentes"], reverse=True)

    por_coluna: list[dict] = []
    for info in linhas:
        col = info["coluna"]
        serie = df[col]
        if pd.api.types.is_numeric_dtype(serie):
            tipo = "numerica"
        elif pd.api.types.is_datetime64_any_dtype(serie):
            tipo = "temporal"
        else:
            tipo = "categorica"
        info["tipo"] = tipo
        info["sugestao_remover"] = info["pct_ausentes"] >= AUSENTES_CRITICO
        info["sugestao_imputar"] = info["pct_ausentes"] >= AUSENTES_ALERTA
        por_coluna.append({
            "coluna": col,
            "tipo": tipo,
            "pct_ausentes": info["pct_ausentes"],
            "sugestao_remover": info["sugestao_remover"],
            "sugestao_imputar": info["sugestao_imputar"],
            "estrategia_imputacao": _sugerir_imputacao(tipo, info["pct_ausentes"]),
        })

    # ajusta status leve se qualquer coluna acima do alerta
    if status_geral == "ok" and any(x["pct_ausentes"] > 0 for x in linhas):
        status_geral = "warn"

    return VerdictAmbientes(status=status_geral, linhas=linhas, percentual_por_coluna=pct_map, por_coluna=por_coluna)


def heatmap_dados_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna um DataFrame binário (0/1) sinalizando onde há ausência."""
    return df.isna().astype(int)