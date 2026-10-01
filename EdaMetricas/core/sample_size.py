"""Adequação do tamanho amostral por tipo de tarefa.

Regras aplicadas:
- Regressão: Regra de Green — n >= 50 + 8k (modelo geral) e
  n >= 104 + k (teste de preditores individuais). k = nº de features numéricas.
- Classificação: mínimo ~20 amostras/classe por feature; alerta se alguma
  classe tiver menos de ~30 exemplos.
- Série Temporal: precisa ao menos de um número mínimo de observações e,
  quanto houver período sazonal, é recomendável ter 2-3 ciclos completos.
- Clusterização: alerta quando n é baixo frente ao nº de dimensões
  (maldição da dimensionalidade).

O resultado é um semáforo (✅/⚠️/❌) atualizado conforme a regra e a tarefa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils.constants import (
    CLASS_MIN_CLASSE_ABSOLUTO,
    CLASS_MIN_POR_CLASSE_FEATURE,
    DESEQUILIBRIO_SEVEO,
    EPSILON,
    REGRA_REG_BASE,
    REGRA_REG_BASE2,
    REGRA_REG_POR_FEATURE,
    REGRA_REG_POR_FEATURE2,
)


@dataclass
class VerdictAmostragem:
    """Resultado do julgamento de tamanho de amostra.

    status: 'ok' | 'warn' | 'error'
    """
    tarefa: str = ""
    status: str = ""
    observacoes: list[str] = field(default_factory=list)
    regra_modelo: str = ""
    regra_preditor: str = ""


def analisar_tamanho_amostral(
    df: pd.DataFrame,
    alvo: Optional[str] = None,
    tarefa: Optional[str] = None,
    tempo_col: Optional[str] = None,
) -> VerdictAmostragem:
    """Gera o veredito de tamanho de amostra conforme o tipo de tarefa.

    Em "auto", avalia todas (regressão, classificação, série temporal,
    clusterização) e concatena os resultados nas observações.
    """
    n = len(df)
    # k = número de colunas numéricas em potencial (features)
    k = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    if alvo is not None and pd.api.types.is_numeric_dtype(df[alvo]):
        # alvo numérico consume uma feature numérica
        k = max(k - 1, 0)

    if n == 0:
        return VerdictAmostragem(
            tarefa=tarefa or "generico",
            status="error",
            observacoes=["Sem observações disponíveis nos dados."],
        )

    if tarefa == "regressao":
        return _verificar_regresso(n, k)
    if tarefa == "classificacao":
        return _verificar_classificacao(df, alvo, k)
    if tarefa == "serie_temporal":
        return _verificar_serie_temporal(df)
    if tarefa == "clusterizacao":
        return _verificar_clusterizacao(n, k)
    # auto / generico -> avalia todas
    return _verificar_auto(df, alvo, k, tempo_col)


def _verificar_regresso(n: int, k: int) -> VerdictAmostragem:
    """Regra de Green para regressão."""
    limite_modelo = REGRA_REG_BASE + REGRA_REG_POR_FEATURE * k
    limite_preditor = REGRA_REG_BASE2 + REGRA_REG_POR_FEATURE2 * k
    observ: list[str] = []

    if n >= limite_modelo:
        observ.append(f"✅ Modelo completo OK: n={n} ≥ {limite_modelo}.")
    else:
        observ.append(f"⚠️ Modelo completo fraco: n={n} < {limite_modelo}.")
    if n >= limite_preditor:
        observ.append(f"✅ Preditores individuais OK: n={n} ≥ {limite_preditor}.")
    else:
        observ.append(f"⚠️ Preditores fracos: n={n} < {limite_preditor}.")

    status = "ok" if n >= limite_modelo else "warn"
    if n < limite_modelo * 0.5:
        status = "error"

    msg = (
        f"Regra de Green: modelo {limite_modelo} / preditor {limite_preditor} "
        f"(n={n}, k={k})."
    ) if status == "ok" else (
        f"Amostra abaixo do mínimo recomendado: n={n} vs "
        f"{limite_modelo} (modelo) / {limite_preditor} (preditor)."
    )
    return VerdictAmostragem(
        tarefa="regressao",
        status=status,
        observacoes=observ + [msg],
        regra_modelo=f"n ≥ 50 + 8·k = {limite_modelo}",
        regra_preditor=f"n ≥ 104 + k = {limite_preditor}",
    )


def _verificar_classificacao(
    df: pd.DataFrame, alvo: Optional[str], k_features: int
) -> VerdictAmostragem:
    """Regras para classificação.

    - ~20 amostras por classe e por feature (recomendação de mercado).
    - Alerta se alguma classe tiver < CLASS_MIN_CLASSE_ABSOLUTO exemplos.
    - Alerta de desbalanceamento quando a razão for alta.
    """
    if alvo is None or alvo not in df.columns:
        n = len(df)
        estim = CLASS_MIN_POR_CLASSE_FEATURE * max(k_features, 1)
        return VerdictAmostragem(
            tarefa="classificacao",
            status="warn",
            observacoes=[
                "Alvo não definido; estimativa para classificação depende do alvo.",
                f"n={n} vs. estimativa n >= {CLASS_MIN_POR_CLASSE_FEATURE}·k "
                f"(k={k_features}) => >= {estim}.",
            ],
        )

    serie = df[alvo]
    serie_ok = serie.dropna()
    # alvo categórico/descrito ou com poucos valores -> classificação
    contagem = serie_ok.astype(str).value_counts()
    n_classes = int(contagem.size)
    if n_classes == 0:
        return VerdictAmostragem(
            tarefa="classificacao",
            status="error",
            observacoes=["Sem observações válidas para classificação."],
        )

    min_classe = int(contagem.min())
    razao = float(contagem.max() / max(min_classe, 1))
    min_recomendado = CLASS_MIN_POR_CLASSE_FEATURE * k_features

    observ: list[str] = []
    observ.append(f"Classes detectadas: {n_classes}.")
    observ.append(f"Menor classe: {min_classe} observações.")
    observ.append(f"Razão de classes (maior/menor): {razao:.1f}:1.")
    observ.append(f"~{CLASS_MIN_POR_CLASSE_FEATURE} amostras/classe × k={k_features} => ≈{min_recomendado}.")

    status = "ok"
    if len(df) < min_recomendado and razao > 1.0:
        observ.append(f"Amostra total abaixo da recomendação ≈{min_recomendado}.")
        status = "warn"
    if min_classe < CLASS_MIN_CLASSE_ABSOLUTO:
        observ.append(f"Classe(s) pequena(s) (< {CLASS_MIN_CLASSE_ABSOLUTO}): alerta para aprendizado.")
        status = "warn"
    if razao >= DESEQUILIBRIO_SEVEO:
        observ.append(f"Desbalanceamento severo ({razao:.1f}:1): considere SMOTE/class_weight/stratificação.")
        status = "warn"
    elif razao >= 3.0:
        observ.append(f"Desbalanceamento leve ({razao:.1f}:1).")
        if status == "ok":
            status = "ok"

    return VerdictAmostragem(
        tarefa="classificacao",
        status=status,
        observacoes=observ,
    )


def _verificar_serie_temporal(df: pd.DataFrame) -> VerdictAmostragem:
    """Verificação genérica do tamanho de amostra para séries temporais.
    """
    n = len(df)
    observ: list[str] = [f"Observações disponíveis: {n}."]
    if n < 30:
        observ.append("Série curta (< 30) faz decomposição/predições frágil.")
        status = "warn"
    elif n < 60:
        observ.append("Série razoável (30–60): migramente utilizável.")
        status = "ok"
    else:
        observ.append(f"Confortável para série temporal (n={n}).")
        status = "ok"
    return VerdictAmostragem(tarefa="serie_temporal", status=status, observacoes=observ)


def _verificar_clusterizacao(n: int, k_features: int) -> VerdictAmostragem:
    """Avaliação para clusterização (curse of dimensionality)."""
    observ: list[str] = []
    if n < 10:
        return VerdictAmostragem(
            tarefa="clusterizacao",
            status="error",
            observacoes=["Amostra muito pequena para clusterização."],
        )
    if 3 * k_features > n:
        observ.append(f"Maldição da dimensionalidade: n={n} < 3·k ({k_features}).")
        status = "warn"
    else:
        observ.append(f"Adequado para clusterização: n={n}, k={k_features}.")
        status = "ok"
    return VerdictAmostragem(tarefa="clusterizacao", status=status, observacoes=observ)


def _verificar_auto(
    df: pd.DataFrame,
    alvo: Optional[str],
    k: int,
    tempo_col: Optional[str],
) -> VerdictAmostragem:
    """Exibe veredito compilado das quatro tarefas em sequência."""
    blocos: list[str] = []
    for nome, v in (
        ("Regressão", _verificar_regresso(len(df), k)),
        ("Classificação", _verificar_classificacao(df, alvo, k)),
        ("Série Temporal", _verificar_serie_temporal(df)),
        ("Clusterização", _verificar_clusterizacao(len(df), k)),
    ):
        blocos.append(f"--- {nome} ---")
        for msg in v.observacoes:
            blocos.append(msg)
    return VerdictAmostragem(tarefa="geral", status="ok", observacoes=blocos)
