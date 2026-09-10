# Pandas vs Polars vs Spark: Benchmark de Performance

Comparação prática de performance entre Pandas, Polars e Apache Spark, executando as mesmas transformações e agregações em diferentes volumes de dados.

## 🎯 Objetivo

Entender, com dados reais de execução, os limites e pontos fortes de cada ferramenta conforme o volume de dados cresce — e quando faz sentido migrar de uma para outra.

## 🧪 Metodologia

- **Dados**: dataset sintético (volumes variados) + dataset público
- **Transformações testadas**: filter, groupby, join, window, sort, string ops, pipeline combinado
- **Execução**: N rodadas por combinação (ferramenta × transformação × volume), cold e warm start
- **Métricas coletadas**: tempo de execução e uso de memória
- **Ambiente**: containers Docker isolados para cada ferramenta (garante reprodutibilidade)

## 📁 Estrutura do repositório

```
├── data/                  # geração de dados sintéticos e download de dataset público
├── scripts/
│   ├── pandas/            # transformações implementadas em Pandas
│   ├── polars/            # transformações implementadas em Polars
│   └── spark/             # transformações implementadas em Spark
├── benchmark/
│   ├── runner.py          # orquestra as execuções e coleta métricas
│   └── results/           # CSVs de resultado: {acao}_{tool}_{volume}.csv
├── analysis/
│   └── results_analysis.ipynb   # análise e gráficos comparativos
├── requirements.txt
└── docker-compose.yml
```

## 🚀 Como rodar

```bash
# subir os ambientes
docker-compose up -d

# rodar o benchmark completo
python benchmark/runner.py

# ver a análise
jupyter notebook analysis/results_analysis.ipynb
```

## 📊 Resultados

_(a preencher conforme os benchmarks forem executados)_

| Transformação | Volume | Pandas | Polars | Spark |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## 🔍 Principais conclusões

_(a preencher ao final do projeto)_

## 🛠️ Stack

Python · Pandas · Polars · PySpark · Docker · Jupyter

## 📝 Contexto

Projeto pessoal de estudo e portfólio.