"""Constantes e limiares centrais do EdaMetricas.

Todos os limiares numéricos (VIF, IQR, desbalanceamento etc.) ficam
centralizados aqui para evitar "números mágicos" espalhados pela lógica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# ---------------------------------------------------------------------------
# Configuração geral de análise
# ---------------------------------------------------------------------------

#: Erro aceitável para arredondamento numérico
EPSILON: Final[float] = 1e-6

#: Nível de significância padrão usado nos testes estatísticos (p-value)
ALPHA: Final[float] = 0.05

#: Máximo de classes para alvo ser tratado como classificação por padrão.
#: Acima disso, a prática de classificação fica inviável.
MAX_CLASSES_ALVO_DEFAULT: Final[int] = 10

#: Observações mínimas para a ferramenta começar a inferir tipos e testes
MIN_OBSERVACOES_PARA_INFERIR: Final[int] = 30

#: Percentual de valores únicos que indica alta cardinalidade (candidato a ID).
#: ex.: se uma coluna tem 95%+ valores únicos dentre N linhas, ela é tratada
#: como ID/ruído textual.
PERCENTUAL_ALTA_CARDINALIDADE: Final[float] = 0.80

#: Limiar de correlação (|r|) para considerar DEPENDÊNCIA FORTE.
COR_DEF_FORTE: Final[float] = 0.70

#: Limiares de VIF (Variance Inflation Factor) para multicolinearidade
VIF_ALERTA: Final[float] = 5.0
VIF_CRITICO: Final[float] = 10.0

#: Multiplicador do IQR para o método de outliers (padrão Tukey)
OUTLIER_IQR_FATOR: Final[float] = 1.5
#: Limiar do Z-score para marcar outlier (padrão 3-sigma)
OUTLIER_Z_LIMITE: Final[float] = 3.0

#: Limiar para skewness para considerar a distribuição altamente assimétrica
SKEWNESS_LIMITE_ALTO: Final[float] = 0.8
#: Limiar do coeficiente de variação (CV) para sinalizar dispersão elevada
CV_LIMITE_ALTO: Final[float] = 0.40

#: Razões de desbalanceamento (classe maior / menor) — dicas ao usuário
DESEQUILIBRIO_ALERTA: Final[float] = 3.0
DESEQUILIBRIO_SEVEO: Final[float] = 5.0

#: Limiares de ausentes (missing) por coluna
AUSENTES_CRITICO: Final[float] = 30.0   #: acima disso, sugerir remoção
AUSENTES_ALERTA: Final[float] = 10.0    #: acima disso, sugerir imputação

#: Regras de dimensionamento amostral para regressão (regra de Green):
#:   n >= 50 + 8k  (modelo geral)
#:   n >= 104 + k  (teste de preditores individuais)
REGRA_REG_BASE: Final[int] = 50
REGRA_REG_POR_FEATURE: Final[int] = 8
REGRA_REG_BASE2: Final[int] = 104
REGRA_REG_POR_FEATURE2: Final[int] = 1

#: Regra para classificação: amostras mínimas por classe por feature,
#: e valor mínimo absoluto por classe para não disparar alertas.
CLASS_MIN_POR_CLASSE_FEATURE: Final[int] = 20
CLASS_MIN_CLASSE_ABSOLUTO: Final[int] = 30

#: Número mínimo de ciclos para considerar periodos sazonais confiáveis
PERIODO_SAZONAL_MIN_CICLOS: Final[int] = 2


# ---------------------------------------------------------------------------
# Identidade visual / paleta de cores
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaletaCores:
    """Cores do tema do EdaMetricas (configurável pela UI)."""

    principal: str = "#1976D2"     # azul
    secundaria: str = "#00ACC1"    # teal/ciano
    alerta: str = "#FFA000"        # amarelo (⚠️)
    erro: str = "#D32F2F"          # vermelho (❌)
    sucesso: str = "#388E3C"       # verde (✅)
    fundo: str = "#F5F5F5"         # cinza claro
    fundo2: str = "#EEEEEE"
    borda: str = "#BDBDBD"


PALETA: Final[PaletaCores] = PaletaCores()