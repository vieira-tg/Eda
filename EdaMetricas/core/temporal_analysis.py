"""Detectar e analisar ordem temporal: monotonia, ADF, ACF/PACF, decomposição."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from utils.constants import ALPHA


@dataclass
class InfoTemporal:
    """Informações de decomposição e estacionariedade da série."""

    periodo_estima: Optional[int] = None
    num_ciclos: Optional[float] = None
    adf_estatistica: float = float("nan")
    adf_p_valor: float = float("nan")
    serie_estacionaria: Optional[bool] = None


@dataclass
class ResultadoTemporal:
    """Resultado da análise temporal de uma série."""

    coluna: str = ""
    n_validos: int = 0
    monotona: Optional[bool] = None
    frequencia: Optional[str] = None
    info: InfoTemporal = field(default_factory=InfoTemporal)
    observacao: Optional[str] = None


def _frequencia_por_datas(datas: pd.Series) -> str:
    """Estima a frequência de uma coluna datetime (mediana dos gaps)."""
    datas = pd.to_datetime(datas.dropna())
    if datas.empty:
        return "desconhecida"
    if not datas.is_monotonic_increasing:
        datas = datas.sort_values()
    diffs = datas.diff().dropna()
    if diffs.empty:
        return "desconhecida"
    days = diffs.median().total_seconds() / 86400.0
    if days <= 0:
        return "irregular"
    if days < 1.5:
        return "diario"
    if days < 3.0:
        return "diario (irregular)"
    if days < 7.9:
        return "semanal"
    if days < 31.0:
        return "mensal"
    if days < 365.0:
        return "trimestral"
    return "anual/maior"


def _estimar_periodo_por_acf(serie: pd.Series, max_lags: int = 50) -> Optional[int]:
    """Estima o período sazonal via ACF (pico significativo após lag 1)."""
    serie = serie.dropna()
    if len(serie) < 8:
        return None
    max_lags = min(max_lags, max(10, len(serie) // 2))
    try:
        acf = sm.stats.stattools.acf(serie.values, nlags=max_lags, fft=False)
    except Exception:
        return None
    candidatos = np.where(np.abs(acf[1:]) > 0.10)[0]
    if len(candidatos) == 0:
        return None
    return int(candidatos.argmax() + 2)  # +2 porque o índice 0 é o próprio dado


def analisar_temporal(
    serie_numerica: pd.Series,
    serie_tempo: Optional[pd.Series] = None,
) -> ResultadoTemporal:
    """Análise temporal e de séries.

    Parâmetros:
        serie_numerica: valores a medir (ex.: preço boi ao longo do tempo).
        serie_tempo: opcional; quando fornecido, ordena a série por tempo e
            determina a frequência.

    Retorna:
        ResultadoTemporal com ADF, ACF/PACF e decomposição estimada.
    """
    res = ResultadoTemporal()
    serie_numerica = serie_numerica.copy()

    # Se houver coluna de tempo, ordenamos por ela (sem embaralhamento)
    if serie_tempo is not None and serie_tempo.notna().any():
        try:
            dados = pd.DataFrame({"valor": serie_numerica, "tempo": serie_tempo})
            dados = dados.dropna(subset=["tempo"]).sort_values("tempo")
            serie_numerica = dados["valor"]
        except Exception:
            # se falha, segue com a série numérica crua
            pass

    serie_numerica = serie_numerica.dropna()
    res.n_validos = int(len(serie_numerica))
    if res.n_validos < 4:
        res.observacao = "Poucos valores válidos (< 4) para análise de série temporal."
        return res

    # ADF: teste de estacionariedade
    try:
        import warnings
        # suprime warnings de design rank-deficient (série perfeitamente correlacionada)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = adfuller(serie_numerica.values, regression="c", result_object=False)
        adf_stat, p_valor = r[0], r[1]
        res.info.adf_estatistica = float(adf_stat)
        res.info.adf_p_valor = float(p_valor)
        res.info.serie_estacionaria = bool(p_valor < ALPHA)
    except Exception:
        res.info.serie_estacionaria = None

    # frequência / período / ciclos
    if serie_tempo is not None and pd.api.types.is_datetime64_any_dtype(serie_tempo):
        res.frequencia = _frequencia_por_datas(serie_tempo)
    res.info.periodo_estima = _estimar_periodo_por_acf(serie_numerica)
    if res.info.periodo_estima and res.info.periodo_estima > 0:
        res.info.num_ciclos = res.n_validos / res.info.periodo_estima

    # monotonia da série já ordenada
    try:
        ordenada = serie_numerica.sort_values()
        res.monotona = bool(ordenada.is_monotonic_increasing or ordenada.is_monotonic_decreasing)
    except Exception:
        res.monotona = None

    return res
