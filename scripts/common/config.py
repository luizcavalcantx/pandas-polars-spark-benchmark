"""
Configuracao compartilhada do benchmark: caminhos dos datasets, lista de
operacoes e parametros de execucao (repeticoes, warm-up).

Ajuste os caminhos aqui se sua estrutura de pastas for diferente -- todos
os outros scripts (pandas/polars/spark/runner) importam deste arquivo.
"""
from pathlib import Path

# scripts/common/config.py -> parents[0]=common, [1]=scripts, [2]=raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "1m": PROJECT_ROOT / "data/synthetic/sales_1m.parquet",
    "10m": PROJECT_ROOT / "data/synthetic/sales_10m.parquet",
    "50m": PROJECT_ROOT / "data/synthetic/sales_50m.parquet",
    "200m": PROJECT_ROOT / "data/synthetic/sales_200m.parquet",
}

DIM_CUSTOMERS = PROJECT_ROOT / "data/synthetic/dim_customers.parquet"
DIM_PRODUCTS = PROJECT_ROOT / "data/synthetic/dim_products.parquet"

OPERATIONS = ["filter", "groupby", "join", "window", "sort", "string_ops", "pipeline"]
TOOLS = ["pandas", "polars", "spark"]

REPEATS = 3       # execucoes cronometradas por combinacao (tool, dataset, operacao)
WARMUP_RUNS = 1   # execucoes de aquecimento antes de cronometrar (nao entram no resultado)

RESULTS_CSV = PROJECT_ROOT / "benchmark/results/results.csv"
