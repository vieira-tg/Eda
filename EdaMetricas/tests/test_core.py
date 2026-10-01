"""Testes rápidos dos módulos de core/ (sem GUI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataLoader
from core.descriptive_stats import analisar_series
from core.missing_values import avaliar_ausentes
from core.correlation import analisar_correlacao
from core.outliers import analisar_outliers
from core.sample_size import analisar_tamanho_amostral
from core.statistical_tests import avaliar_testes_estatisticos
from core.target_analysis import analisar_alvo
from core.temporal_analysis import analisar_temporal
from core.verdict_engine import veredito_geral, VereditoGeral


def _dataset_regressao(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Dataset sintético para verificar regressão."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 3, n)
    x2 = rng.normal(5, 2.5, n)
    ruido = rng.normal(0, 2.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 3 * x1 + 1.5 * x2 + ruido})


def _dataset_classificacao(n: int = 120, seed: int = 7, seed_bin: int = 13) -> pd.DataFrame:
    """Dataset binário sintético."""
    rng = np.random.default_rng(seed)
    n = n
    x = np.linspace(0, 1, n)
    xy = rng.normal(0, 0.1, n)
    classe = np.where(x + xy > 0.5, "alto", "baixo")  # classes aproximadamente equilibradas
    return pd.DataFrame({
        "f1": np.random.default_rng(seed_bin).normal(0, 1, n),
        "f2": np.random.default_rng(seed_bin + 1).normal(0, 1, n),
        "classe": classe,
    })


def _dataset_serie(n: int = 100) -> pd.DataFrame:
    """Dataset temporal com sazonalidade."""
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    serie = np.sin(2 * np.pi * t / 14.0) + np.random.default_rng(1).normal(0, 0.1, n)
    return pd.DataFrame({"data": idx, "valor": serie})


class TestLoader:
    def test_inferencia_basica(self):
        df = _dataset_regressao()
        loader = DataLoader()
        perfil = loader.perfil(df, caminho="<sintetico>")
        assert perfil.linhas == 200
        assert "y" in perfil.colunas_numericas


class TestEstatisticas:
    def test_estatisticas_regressao(self):
        df = _dataset_regressao()
        s = analisar_series(df)
        assert "y" in s
        assert abs(s["y"]["media"] - df["y"].mean()) < 1e-6


class TestAusentes:
    def test_detecta_ausentes(self):
        df = _dataset_regressao()
        df.loc[0:5, "x1"] = np.nan
        r = avaliar_ausentes(df)
        assert any(x["pct_ausentes"] > 0 for x in r.linhas)
        n, p = df["x1"].isna().sum(), 200 // 100
        # confirma % correto
        assert abs((r.linhas[0]["pct_ausentes"] / 100.0) - (df["x1"].isna().mean())) < 1e-6


class TestCorrelacao:
    def test_pearson_presente(self):
        df = _dataset_regressao()
        res = analisar_correlacao(df, features=["x1", "x2"], alvo="y")
        assert "pearson" in res
        assert not res["pearson"].empty


class TestOutliers:
    def test_detecta_outlier(self):
        df = _dataset_regressao()
        df.loc[0, "x1"] = 1_000.0  # outlier forçado
        res = analisar_outliers(df)
        assert "por_coluna" in res
        assert (res["por_coluna"]["x1"]["iqr"]["total"] > 0) or (
            res["por_coluna"]["x1"]["z_score"]["total"] > 0
        )


class TestSampleSize:
    def test_regressao(self):
        df = _dataset_regressao()
        r = analisar_tamanho_amostral(df, alvo="y", tarefa="regressao")
        assert r.status in ("ok", "warn", "error")


class TestTargetAnalysis:
    def test_regressao(self):
        df = _dataset_regressao()
        r = analisar_alvo(df, alvo="y", forca_tarefa="regressao")
        assert r.tipo == "regressao"

    def test_classificacao(self):
        df = _dataset_classificacao()
        r = analisar_alvo(df, alvo="classe", forca_tarefa="classificacao")
        assert r.tipo == "classificacao"


class TestStatisticalTests:
    def test_basico(self):
        df = _dataset_classificacao()
        r = avaliar_testes_estatisticos(df, alvo="classe")
        assert isinstance(r.itens, list)


class TestTemporal:
    def test_monotona(self):
        df = _dataset_serie()
        res = analisar_temporal(df["valor"], serie_tempo=pd.to_datetime(df["data"]))
        assert res.n_validos == len(df)


class TestVerdictEngine:
    def test_regressao(self):
        df = _dataset_regressao()
        res = veredito_geral(df, alvo="y", tarefa="regressao")
        assert isinstance(res, VereditoGeral)
        # sinais semânticos
        assert res.reg.status in ("ok", "warn", "error")
        assert res.cls.status in ("ok", "warn", "error")
        assert res.ser.status in ("ok", "warn", "error")
        assert res.clu.status in ("ok", "warn", "error")
