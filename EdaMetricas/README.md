# EdaMetricas — Agente de EDA Genérico e Interativo (Python + PySide6)

Aplicativo desktop de Análise Exploratória de Dados totalmente genérico:
carrega qualquer dataset tabular (CSV, XLSX, Parquet, JSON) e, com o usuário
selecionando colunas interativamente, emite um veredicto automático sobre a
adequação do dataset para **Regressão, Classificação, Clusterização ou
Série Temporal**.

## Instalação

```bash
cd Eda
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r EdaMetricas\requirements.txt
.venv\Scripts\python.exe EdaMetricas\main.py
```

O venv fica em `Eda\.venv` e o código-fonte do app em `Eda\EdaMetricas\`.

## Como usar

1. Clique em **Abrir Dataset** e escolha um `.csv`, `.xlsx`, `.parquet` ou
   `.json`. Alternativamente, carregue um **dataset de exemplo** com o
   combo + botão **Carregar Exemplo** (cenários: Regressão, Classificação,
   Série Temporal, Desbalanceado, Com ausências).
2. O painel lateral mostra os tipos de coluna, o alvo recomendado e um
   selector de tarefa (`Auto-detectar / Regressão / Classificação /
   Clusterização / Série Temporal`).
3. Clique em **Analisar** e navegue pelas abas para ver: Visão Geral,
   Estatísticas, Ausentes, Alvo, Correlação, Outliers, Temporal e Veredito.

## O que o app calcula (com as fórmulas)

### Visão Geral
- Shape, uso de memória, duplicatas (absoluto e percentual), colunas com
  alta cardinalidade (suspeitas de serem IDs) e colunas constantes.

### Estatísticas Descritivas
- Numéricas: média, mediana, moda, desvio padrão, variância, mínimo, máximo,
  amplitude, Q1/Q3, IQR, coeficiente de variação, skewness (Fisher-Pearson),
  curtose (excesso).
- Categoricas: frequência absoluta/relativa, nº de categorias únicas, moda.
- Datetime: mínimo/máximo, período entre dados (frequência), quantidade.

### Tamanho Amostral (veredito semáforo)
- **Regressão**: Regra de Green — `n ≥ 50 + 8k` (modelo geral) e
  `n ≥ 104 + k` (teste de preditores individuais), `k = número de features`.
- **Classificação**: mínimo recomendado de ~20 amostras/classe por feature;
  alerta se alguma classe tiver < 30 exemplos.
- **Série Temporal**: alerta se a série não tiver ao menos ~2 ciclos de um
  período estimado ou se houver poucos pontos (< 30/60 conforme contexto).
- **Clusterização**: maldição da dimensionalidade quando `n < 3·k`.

### Valores Ausentes
- % de ausentes por coluna + limiares fixos (`AUSENTES_ALERTA=10%`,
  `AUSENTES_CRITICO=30%`) marcando remoção/tenção.
- Estratégia de imputação recomendada por tipo de variável (média/mediana
  para numérica, moda para categórica, ffill/interpolação para temporal).

### Correlação e Multicolinearidade
- **Pearson** (linear) e **Spearman** (monotônica) para pares numéricas.
- **Cramér's V** para pares categóricas.
- **ANOVA / eta-squared** para categórico → numérico.
- **VIF** (Variance Inflation Factor) dos numéricos (limiares
  `5` = alerta, `10` = crítico).
- Ranking de features vs alvo (quando alvo definido).

### Outliers
- **Método IQR** (Q1−1.5·IQR / Q3+1.5·IQR) e **Z-score** (|z|>3).
- Opcional: Isolation Forest (detecção multivariada).

### Testes Estatísticos
- Normalidade: Shapiro-Wilk (n < 5000), KS/D'Agostino (quando aplicável).
- Homocedasticidade: Levene.
- Comparação de grupos (ANOVA/Kruskal-Wallis conforme simetria).
- Chi² de aderência à distribuição uniforme (para categóricas).

### Análise da Variável Alvo
- Detecção automática se o alvo é contínuo (→ regressão) ou categórico/
  discreto com baixa cardinalidade (→ classificação); override manual.
- Regressão: histograma, boxplot, teste de normalidade + transformação
  recomendada conforme skewness.
- Classificação: contagem/proporção por classes, razão desbalanceamento,
  sugereções de SMOTE/class_weight/como resolver.

### Análise Temporal
- Detecção automática de colunas datetime.
- Teste **ADF** (augmented Dickey-Fuller) → estabilidade da série.
- ACF/PACF plotadas ( quando existir a coluna de tempo ).
- Decomposição sazonal (tendência/sazonalidade/resíduo) com 2+ ciclos.

### Veredicto Automático
Consolidado para as 4 tarefas (Regressão, Classificação, Clusterização e
Série Temporal), com o pior sinal de cada análise carregando o œselo. Cada
verdito vem acompanhado das razões (ex.: "Inadequado para série temporal:
nenhuma coluna datetime detectada").

## Estrutura do projeto

```
EdaMetricas/
├── main.py              # ponto de entrada (QApplication + MainWindow)
├── requirements.txt     # dependências fixadas
├── README.md            # este arquivo
├── core/
│   ├── data_loader.py      # leitura + inferência de tipos
│   ├── descriptive_stats.py
│   ├── sample_size.py      # adequação de tamanho amostral
│   ├── missing_values.py
│   ├── correlation.py      # corr + VIF + ranking vs alvo
│   ├── outliers.py
│   ├── statistical_tests.py
│   ├── target_analysis.py
│   └── verdict_engine.py   # consolida tudo em parecer final
├── ui/
│   ├── main_window.py      # janela principal (QTabWidget por análise)
│   └── widgets/
│       ├── column_selector.py  # painel lateral de seleção
│       ├── chart_canvas.py     # wrapper de matplotlib + toolbar
│       └── summary_table.py    # tabela de resumo ordenável
├── utils/
│   ├── constants.py        # limiares (VIF, IQR, desbalanceamento etc.)
│   └── formatting.py       # helpers de formatação numérica/temporal
└── tests/
    └── test_core.py        # pytest para cada módulo de core/
```

## Boas Práticas

- **Separação core/GUI**: toda a lógica estatística vive em `core/` (sem
  dependência de PySide6). A GUI só chama essas funções.
- **Parâmetros centralizados**: todos os limiares (`VIF_CRITICO`, `IQR`,
  `AUSENTES_CRITICO` etc.) ficam em `utils/constants.py`.
- **Type hints** e docstrings em todo código público.
- **Tratamento de erros**: falhas ao abrir/analisar são mostradas ao
  usuário via `QMessageBox` (nada trava o app).
- **Testes**: `pytest` rodando `tests/test_core.py` cobre os principais
  pipelines de `core/`.

## Testes

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

Todos os testes passam (11 testes cobrindo os principais componentes core).
