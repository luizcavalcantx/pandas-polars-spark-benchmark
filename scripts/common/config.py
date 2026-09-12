"""
Shared benchmark configuration: dataset paths, list of
operations and execution parameters (repeats, warm-up).

Adjust the paths here if your folder structure is different -- all
other scripts (pandas/polars/spark/runner) import from this file.
"""
from pathlib import Path

# scripts/common/config.py -> parents[0]=common, [1]=scripts, [2]=project root
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

REPEATS = 3       # timed runs per combination (tool, dataset, operation)
WARMUP_RUNS = 1   # warm-up runs before timing (not included in the result)

RESULTS_CSV = PROJECT_ROOT / "benchmark/results/results.csv"
