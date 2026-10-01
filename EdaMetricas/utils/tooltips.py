"""Textos de ajuda (tooltips) do EdaMetricas.

Cada chave é o nome de um elemento da UI ou de uma métrica, e o valor é a
explicação curta exibida no balão ao passar o mouse ou clicar.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Botões e controles da janela principal
# ---------------------------------------------------------------------------

BOTOES: Final[dict[str, str]] = {
    "abrir": (
        "Abre um arquivo de dados tabulares (CSV, XLSX, Parquet ou JSON). "
        "O app detecta automaticamente o tipo de cada coluna."
    ),
    "carregar_exemplo": (
        "Carrega um dataset sintético de demonstração, escolhido no combo ao "
        "lado. Ideal para conhecer o fluxo do app sem precisar de um arquivo."
    ),
    "analisar": (
        "Roda a bateria completa de análises (estatísticas, ausentes, "
        "correlação, outliers, testes, temporal e veredito) sobre o dataset "
        "carregado e preenche todas as abas."
    ),
    "exportar": (
        "Gera um relatório HTML autocontido com tabelas, gráficos e o "
        "veredito final, pronto para anexar ao relatório do trabalho."
    ),
    "combo_exemplo": (
        "Escolha um dataset sintético: Regressão, Classificação, Série "
        "Temporal, Desbalanceado, Com ausências, Multicolinear, "
        "Não estacionária ou Completo (preenche todas as abas)."
    ),
}

# ---------------------------------------------------------------------------
# Painel lateral (seleção de colunas)
# ---------------------------------------------------------------------------

SELETOR: Final[dict[str, str]] = {
    "alvo": (
        "Coluna alvo (target): a variável que você quer prever. "
        "Contínua (preço, quantidade) → regressão; categorias → classificação."
    ),
    "tempo": (
        "Coluna de data/tempo. Necessária para a análise de séries temporais "
        "(ADF, periodicidade, ACF/PACF). Deixe em branco se não existir."
    ),
    "tarefa": (
        "Tipo de tarefa de ML pretendida. 'Auto-detectar' deixa o app "
        "inferir pelo tipo do alvo. A escolha influencia o veredito final."
    ),
    "features": (
        "Marque as colunas que entrarão como atributos (features) da análise. "
        "Colunas ID/alta cardinalidade e constantes normalmente ficam fora."
    ),
}

# ---------------------------------------------------------------------------
# Abas
# ---------------------------------------------------------------------------

ABAS: Final[dict[str, str]] = {
    "Visão Geral": "Perfil do dataset: shape, memória, tipos de coluna, duplicatas e colunas constantes.",
    "Estatísticas": "Estatísticas descritivas completas por coluna (média, mediana, desvio, skewness, curtose...).",
    "Ausentes": "Percentual de valores ausentes por coluna, candidatas a remoção e estratégia de imputação.",
    "Alvo": "Distribuição da variável alvo: histograma/barras, teste de normalidade, balanceamento de classes.",
    "Correlação": "Matriz de correlação (Pearson/Spearman), VIF de multicolinearidade e ranking vs. alvo.",
    "Outliers": "Detecção de valores anômalos pelos métodos IQR, Z-score e Isolation Forest.",
    "Temporal": "Estacionariedade (teste ADF), frequência estimada, período sazonal e monotonicidade.",
    "Veredito": "Parecer automático de adequação do dataset para Regressão, Classificação, Clusterização e Série Temporal.",
}

# ---------------------------------------------------------------------------
# Métricas (cabeçalhos de tabela) — chave em minúsculas
# ---------------------------------------------------------------------------

METRICAS: Final[dict[str, str]] = {
    # visão geral / estatísticas
    "coluna": "Nome da coluna no dataset.",
    "tipo": "Tipo inferido: numérica, categórica ou temporal.",
    "media": "Média aritmética: soma dos valores dividida pelo nº de observações.",
    "mediana": "Mediana: valor central quando os dados estão ordenados (robusta a outliers).",
    "std": "Desvio-padrão: quanto os valores se afastam, em média, da média.",
    "skewness": "Assimetria da distribuição. ~0 = simétrica; grande |valor| (>0,8) indica cauda pesada — considere transformação (log/Box-Cox).",
    "kurtosis": "Curtose (excesso): peso das caudas. 0 ≈ normal; valores altos indicam caudas grossas (mais extremos).",
    "kurtose": "Curtose (excesso): peso das caudas. 0 ≈ normal; valores altos indicam caudas grossas (mais extremos).",
    "iqr": "Intervalo interquartil (Q3 − Q1): amplitude dos 50% centrais. Base do método de Tukey para outliers.",
    "cv": "Coeficiente de variação (std/média): dispersão relativa. Valores altos indicam grande variabilidade relativa.",
    # ausentes
    "ausentes %": "Percentual de valores ausentes (vazios/NaN) na coluna. > 30% sugere remoção; > 10% sugere imputação.",
    "remover": "Sugestão de remover a coluna quando há ausência excessiva (acima de 30%).",
    "estratégia": "Estratégia sugerida de imputação conforme o tipo (média/mediana, moda ou interpolação temporal).",
    # correlação
    "feature": "Nome da feature (coluna de atributo).",
    "vif": "Variance Inflation Factor: mede multicolinearidade entre features. VIF > 5 = atenção; > 10 = problema.",
    "pearson": "Correlação de Pearson: dependência linear entre −1 e +1.",
    "spearman": "Correlação de Spearman (ranks): dependência monotônica, robusta a não-normalidade.",
    "score": "Força da associação da feature com o alvo (quanto maior, mais informativa).",
    # outliers
    "z-score": "Outliers por Z-score: pontos a mais de 3 desvios-padrão da média.",
    "z": "Outliers por Z-score: pontos a mais de 3 desvios-padrão da média.",
    # temporal
    "adf": "Teste ADF (Augmented Dickey-Fuller): hipótese de raiz unitária. p < 0,05 → série estacionária (boa para prever em nível).",
    "acf": "Autocorrelação: quanto um valor depende dos anteriores (lags).",
    "pacf": "Autocorrelação parcial: dependência direita de um lag, já descontando os lags intermediários.",
    # veredito
    "tarefa": "Tipo de tarefa de machine learning avaliada.",
    "status": "Adequação do dataset: ok (adequado), warn (com ressalvas), error (inadequado).",
    "selo": "Selo visual: ✅ adequado, ⚠️ ressalvas, ❌ inadequado.",
    "razoes": "Motivos que motivaram o veredito (ex.: amostra insuficiente, desbalanceamento, série não estacionária).",
    # alvo
    "classe": "Categoria/classe da variável alvo.",
    "abs": "Contagem absoluta de observações da classe.",
    "pct": "Percentual da classe sobre o total.",
}


def tooltip_metrica(nome: str) -> str:
    """Devolve a explicação para o nome de métrica informado (ou vazio).

    A busca é feita em minúsculas para tolerar variações de capitalização.
    """
    return METRICAS.get(nome.strip().lower(), "")
