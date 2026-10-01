"""Carregamento do dataset + inferência de tipo de coluna + perfil geral.

Pensado para aceitar qualquer CSV/XLSX/Parquet/JSON, sem lógica hardcoded
para um domínio específico. Tudo é derivado do próprio DataFrame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.constants import (
    PERCENTUAL_ALTA_CARDINALIDADE,
)

logger = logging.getLogger(__name__)


@dataclass
class PerfilDataset:
    """Resumo geral do dataset após o carregamento."""

    caminho: str = ""
    linhas: int = 0
    colunas: int = 0
    memoria_mb: float = 0.0
    # dicionário coluna -> tipo inferido
    tipos: dict[str, str] = field(default_factory=dict)
    # lista de nomes de colunas de cada tipo
    colunas_numericas: list[str] = field(default_factory=list)
    colunas_categoricas: list[str] = field(default_factory=list)
    colunas_temporais: list[str] = field(default_factory=list)
    colunas_id: list[str] = field(default_factory=list)  # alta cardinalidade
    colunas_constantes: list[str] = field(default_factory=list)
    # qualidade
    n_duplicatas: int = 0
    percentual_duplicatas: float = 0.0


class DataLoader:
    """Carrega o dataset e infere tipos de coluna de forma genérica."""

    EXTENSOES_SUPORTADAS = {".csv", ".xlsx", ".xls", ".parquet", ".json"}

    def carregar(self, caminho: str | Path) -> pd.DataFrame:
        """Lê o arquivo e devolve um DataFrame bruto (sem tocar nos dados)."""
        caminho = Path(caminho)
        ext = caminho.suffix.lower()
        if ext not in self.EXTENSOES_SUPORTADAS:
            raise ValueError(f"Extensão não suportada: {ext}")

        logger.info("Carregando dataset: %s", caminho)
        if ext == ".csv":
            df = pd.read_csv(caminho)
        elif ext == ".xlsx":
            df = pd.read_excel(caminho, engine="openpyxl")
        elif ext == ".xls":
            df = pd.read_excel(caminho)
        elif ext == ".parquet":
            df = pd.read_parquet(caminho)
        elif ext == ".json":
            df = pd.read_json(caminho)
        else:
            raise ValueError(f"Extensão não suportada: {ext}")
        return df

    # ---- inferência de tipos -------------------------------------------------

    def _inferir_tipos(self, df: pd.DataFrame) -> dict[str, str]:
        """Classifica cada coluna em: numerica | categorica | temporal | id.

        Regras simples e genéricas:
        - datetime já vindo do pandas → temporal
        - coluna numérica → numerica
        - object/category/bool → categorica SE não for alta-cardinalidade,
          senão vira 'id' (provável identificador/texto livre)
        """
        tipos: dict[str, str] = {}
        n = max(len(df), 1)
        for col in df.columns:
            serie = df[col]
            if pd.api.types.is_datetime64_any_dtype(serie):
                tipos[col] = "temporal"
            elif pd.api.types.is_bool_dtype(serie):
                tipos[col] = "categorica"
            elif pd.api.types.is_numeric_dtype(serie):
                tipos[col] = "numerica"
            elif isinstance(serie.dtype, pd.CategoricalDtype):
                tipos[col] = "categorica"
            else:
                # object ou string: tenta virar datetime primeiro
                if self._parece_datetime(serie):
                    tipos[col] = "temporal"
                else:
                    n_unique = serie.nunique(dropna=True)
                    if (n_unique / n) >= PERCENTUAL_ALTA_CARDINALIDADE and n_unique > 50:
                        tipos[col] = "id"
                    else:
                        tipos[col] = "categorica"
        return tipos

    def _parece_datetime(self, serie: pd.Series) -> bool:
        """Heurística barata: tenta converter amostra não-nula para datetime."""
        amostra = serie.dropna().astype(str).head(30)
        if len(amostra) == 0:
            return False
        # precisa ter padrão típico de data (ex.: contém '-' ou '/')
        if not amostra.str.contains(r"[-/]").mean() > 0.8:
            return False
        try:
            pd.to_datetime(amostra, errors="raise")
            return True
        except Exception:
            return False

    # ---- perfil geral --------------------------------------------------------

    def perfil(self, df: pd.DataFrame, caminho: str = "") -> PerfilDataset:
        """Gera o PerfilDataset do DataFrame carregado."""
        tipos = self._inferir_tipos(df)
        perfil = PerfilDataset(
            caminho=caminho,
            linhas=len(df),
            colunas=df.shape[1],
            memoria_mb=round(df.memory_usage(deep=True).sum() / (1024**2), 3),
            tipos=tipos,
        )
        for col, tipo in tipos.items():
            if tipo == "numerica":
                perfil.colunas_numericas.append(col)
            elif tipo == "categorica":
                perfil.colunas_categoricas.append(col)
            elif tipo == "temporal":
                perfil.colunas_temporais.append(col)
            elif tipo == "id":
                perfil.colunas_id.append(col)

        # colunas constantes (candidatas a exclusão)
        perfil.colunas_constantes = [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ]

        perfil.n_duplicatas = int(df.duplicated().sum())
        perfil.percentual_duplicatas = (
            perfil.n_duplicatas / perfil.linhas if perfil.linhas else 0.0
        )
        return perfil

    def carregar_e_perfil(self, caminho: str | Path) -> tuple[pd.DataFrame, PerfilDataset]:
        """Carrega o arquivo e já devolve (DataFrame, Perfil)."""
        df = self.carregar(caminho)
        return df, self.perfil(df, caminho=str(caminho))


def carregar_dataset(caminho: str | Path) -> tuple[pd.DataFrame, PerfilDataset]:
    """Atalho funcional: carrega e retorna (DataFrame, PerfilDataset)."""
    return DataLoader().carregar_e_perfil(caminho)