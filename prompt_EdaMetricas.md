# PROMPT — EdaMetricas: Agente de EDA Genérico e Interativo (Python + PySide6)

## PAPEL
Você é um engenheiro de software Python sênior, especialista em Data Science,
Estatística Aplicada e desenvolvimento de aplicações desktop com PySide6.
Sua tarefa é construir o **EdaMetricas**, um aplicativo desktop de Análise
Exploratória de Dados (EDA) totalmente genérico, capaz de carregar QUALQUER
dataset tabular e, com o usuário selecionando colunas interativamente,
determinar se os dados são adequados para Regressão, Classificação,
Clusterização ou Série Temporal.

## 1. OBJETIVO GERAL
Construir uma ferramenta visual e interativa que:
1. Carregue um dataset (CSV, XLSX, Parquet ou JSON).
2. Permita ao usuário selecionar, via interface gráfica: colunas de features,
   coluna alvo (target), coluna de data/tempo (se existir) e o tipo de tarefa
   de ML pretendida (ou deixar em "auto-detectar").
3. Execute uma bateria completa de análises estatísticas e produza um
   **veredito automático** sobre a adequação do dataset para cada tipo de tarefa.
4. Apresente tudo em gráficos ricos e interativos, com boa identidade visual
   (não o estilo cru padrão do matplotlib).

O app deve funcionar igualmente bem para um dataset de 50 linhas ou de 5 milhões,
sem nenhuma lógica hardcoded para um domínio específico.

## 2. STACK TÉCNICA OBRIGATÓRIA
- Python 3.11+
- **PySide6** — interface gráfica
- **pandas** — manipulação de dados
- **numpy** — cálculos numéricos
- **scipy.stats** — testes estatísticos (Shapiro-Wilk, KS, Chi², ANOVA, etc.)
- **statsmodels** — ADF (estacionariedade), decomposição sazonal, VIF, ACF/PACF
- **scikit-learn** — apenas utilitários (IsolationForest, StandardScaler p/ checagem,
  LabelEncoder), NÃO para treinar modelos — o app é só de diagnóstico
- **matplotlib** (+ **seaborn** opcional para estética) — todos os gráficos,
  embutidos via `FigureCanvasQTAgg`
- **openpyxl** — suporte a Excel
- **missingno** (opcional) — matriz de valores ausentes

## 3. ARQUITETURA DO PROJETO
```
EdaMetricas/
├── main.py
├── requirements.txt
├── README.md              # em português, com instruções de uso
├── core/
│   ├── data_loader.py      # leitura + inferência de tipos de coluna
│   ├── descriptive_stats.py
│   ├── sample_size.py      # adequação de tamanho amostral
│   ├── temporal_analysis.py
│   ├── target_analysis.py  # regressão vs classificação
│   ├── missing_values.py
│   ├── correlation.py
│   ├── outliers.py
│   ├── statistical_tests.py
│   └── verdict_engine.py   # consolida tudo em um parecer final
├── ui/
│   ├── main_window.py
│   └── widgets/
│       ├── column_selector.py
│       ├── chart_canvas.py
│       └── summary_table.py
├── utils/
│   ├── constants.py         # limiares (VIF, IQR, imbalance, etc.) centralizados
│   └── formatting.py
└── tests/
    └── test_*.py            # pytest para cada módulo de core/
```

## 4. FLUXO DE USO (UX)
1. Usuário clica em "Abrir Dataset" → `QFileDialog`.
2. App detecta automaticamente o tipo de cada coluna (numérica, categórica,
   datetime, texto/ID de alta cardinalidade) e mostra um resumo geral.
3. Painel lateral com checkboxes para escolher features, um combobox para
   escolher a coluna alvo, um combobox opcional para coluna temporal, e um
   seletor de tarefa: `Auto-detectar / Regressão / Classificação /
   Clusterização / Série Temporal`.
4. Botão "Analisar" dispara todos os módulos de `core/`.
5. Resultado aparece em `QTabWidget` com uma aba por categoria de análise
   (ver seção 5) + uma aba final "Veredito".

## 5. REQUISITOS FUNCIONAIS DETALHADOS

### 5.1 Carregamento e perfil geral
- Shape (linhas x colunas), uso de memória, tipos inferidos por coluna.
- Contagem de duplicatas, colunas constantes, colunas de alta cardinalidade
  (candidatas a ID, devem ser sinalizadas e sugeridas para exclusão).

### 5.2 Estatísticas descritivas completas (por coluna numérica)
Média, mediana, moda, desvio padrão, variância, mínimo, máximo, amplitude,
Q1/Q3, IQR, coeficiente de variação, assimetria (skewness) e curtose (kurtosis).
Para colunas categóricas: frequência absoluta/relativa, nº de categorias únicas,
moda.

### 5.3 Adequação do tamanho amostral
Aplicar regras de mercado e reportar como um semáforo (✅/⚠️/❌):
- **Regressão**: regra de Green — `n ≥ 50 + 8k` (teste geral do modelo) e
  `n ≥ 104 + k` (teste de preditores individuais), onde k = nº de features.
- **Classificação**: mínimo recomendado de ~10-20 amostras por classe por
  feature; alertar se alguma classe tiver menos de ~30 exemplos.
- **Série Temporal**: exigir pelo menos 2-3 ciclos completos do maior período
  sazonal identificado para permitir decomposição confiável.
- **Clusterização**: alertar se `n` for pequeno demais frente ao nº de
  dimensões (curse of dimensionality).

### 5.4 Detecção e análise de ordem temporal
- Detectar automaticamente colunas datetime (por tipo ou parsing heurístico).
- Verificar se a coluna está ordenada monotonicamente; se não, alertar.
- Calcular frequência/periodicidade (diária, mensal, etc.).
- Teste de estacionariedade **ADF** (`statsmodels.tsa.stattools.adfuller`).
- Autocorrelação (ACF) e autocorrelação parcial (PACF), plotadas.
- Decomposição sazonal (tendência/sazonalidade/resíduo) quando houver dados
  suficientes.

### 5.5 Análise da variável alvo
Detectar automaticamente se o alvo é contínuo (→ regressão) ou categórico/discreto
com baixa cardinalidade (→ classificação); permitir override manual.

**Se regressão:**
- Histograma + KDE, boxplot, Q-Q plot.
- Testes de normalidade: Shapiro-Wilk (n pequeno) e D'Agostino K² / Kolmogorov-Smirnov (n grande).
- Se assimetria alta, sugerir transformação (log, Box-Cox, Yeo-Johnson).

**Se classificação:**
- Contagem e proporção por classe (bar chart).
- Razão de desbalanceamento = classe majoritária / classe minoritária.
- Sinalizar desbalanceamento severo (ex.: minoritária < 10%) e sugerir
  técnicas (SMOTE, class_weight, estratificação no split).
- Teste Chi² de aderência à distribuição uniforme.

### 5.6 Valores ausentes
- % de ausentes por coluna, tabela ordenada + heatmap/matriz visual.
- Sinalizar colunas acima de um limiar configurável (ex.: >30%) como
  candidatas a remoção.
- Sugerir estratégia de imputação por tipo de dado (média/mediana para
  numéricas, moda para categóricas, interpolação para séries temporais).

### 5.7 Correlação e multicolineaasfasridade
- Matriz de correlação **Pearson** (linear) e **Spearman** (monotônica/não-normal)
  entre numéricas, como heatmap.
- **Cramér's V** para pares categórico-categórico.
- **ANOVA / eta-squared** para relação categórica-numérica.
- **VIF** (Variance Inflation Factor) via statsmodels para detectar
  multicolinearidade entre features (VIF > 5-10 = alerta).
- Ranking de correlação de cada feature com o alvo.

### 5.8 Outliers
- Método IQR (limites Q1 − 1.5·IQR e Q3 + 1.5·IQR) e Z-score (|z| > 3),
  aplicados a todas as colunas numéricas.
- Boxplots por coluna.
- Opcional: detecção multivariada com Isolation Forest.

### 5.9 Testes estatísticos formais
Consolidar num painel: normalidade (Shapiro/KS), homocedasticidade (Levene),
ANOVA/Kruskal-Wallis para comparar grupos, Chi² de independência.

### 5.10 Veredito automático (o coração do app)
Um `verdict_engine.py` que agrega os resultados de todos os módulos acima e
gera um parecer textual + selo visual (✅ Adequado / ⚠️ Adequado com ressalvas /
❌ Inadequado) para cada tipo de tarefa:
- Regressão
- Classificação
- Clusterização
- Série Temporal

Cada veredito deve vir acompanhado das razões (ex.: "❌ Inadequado para série
temporal: nenhuma coluna datetime detectada" ou "⚠️ Classificação com
ressalvas: desbalanceamento severo (razão 15:1) — considere reamostragem").

## 6. REQUISITOS DE INTERFACE (GUI)
- Janela principal com título **"EdaMetricas"** e menu/toolbar ("Abrir Dataset",
  "Exportar Relatório").
- Painel lateral fixo com seleção de colunas/target/tarefa.
- `QTabWidget` com uma aba por análise (Visão Geral, Estatísticas, Ausentes,
  Alvo, Correlação, Outliers, Temporal, Veredito).
- Gráficos com identidade visual consistente (paleta de cores definida,
  não o estilo padrão cru do matplotlib) — tema claro/escuro é um diferencial bem-vindo.
- Barra de status com shape do dataset e tempo de processamento.
- Todos os gráficos com toolbar de zoom/pan/salvar (padrão do
  `NavigationToolbar2QT`).

## 7. BOAS PRÁTICAS DE ENGENHARIA (OBRIGATÓRIAS)
- Separação total entre `core/` (lógica pura, testável, sem PySide6) e `ui/`.
- Type hints em todas as funções públicas.
- Docstrings e comentários **em português**, especialmente nos blocos de
  cálculo estatístico — cada etapa relevante (ex.: cálculo de IQR, teste de
  Shapiro) deve ter um comentário curto explicando o "porquê", não só o "o quê".
- Usar pandas/numpy/scipy nativamente para os cálculos (performance e
  corretude), mas sem abrir mão da clareza didática nos comentários.
- Tratamento de erros com try/except e feedback ao usuário via `QMessageBox`
  (nunca deixar a aplicação travar com dataset malformado).
- Uso de `logging` em vez de `print`.
- Constantes/limiares (VIF, IQR multiplier, threshold de imbalance, etc.)
  centralizados em `utils/constants.py`, nunca "números mágicos" espalhados.
- Testes unitários (`pytest`) para os módulos de `core/`, usando datasets
  sintéticos pequenos.
- `requirements.txt` com versões fixadas.

## 8. CRITÉRIOS DE ACEITE (checklist final antes de entregar)
- [ ] Funciona com qualquer CSV/XLSX arbitrário, sem nenhuma coluna hardcoded.
- [ ] Todas as estatísticas descritivas da seção 5.2 aparecem para cada coluna numérica.
- [ ] Veredito de adequação de tamanho amostral é calculado e exibido.
- [ ] Coluna temporal é detectada automaticamente (quando existir) e testada com ADF.
- [ ] Distribuição do alvo é analisada corretamente tanto para regressão quanto
      classificação, com os gráficos e testes corretos para cada caso.
- [ ] Matriz de correlação e VIF funcionam mesmo com colunas mistas (numéricas/categóricas).
- [ ] App não quebra com valores ausentes, colunas constantes ou datasets pequenos (<30 linhas).
- [ ] Veredito final consolidado é exibido para os 4 tipos de tarefa.

## 9. ENTREGÁVEIS
- Código-fonte completo e executável (`python main.py`), no projeto **EdaMetricas**.
- `README.md` em português explicando instalação, uso e critérios usados
  em cada análise (com as fórmulas/regras aplicadas).
- `requirements.txt`.
