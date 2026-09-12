# Pandas vs Polars vs Spark: Performance Benchmark

Practical performance comparison between Pandas, Polars and Apache Spark, running the same transformations and aggregations on different data volumes.

## 🎯 Objective

Understand, with real execution data, the limits and strengths of each tool as data volume grows — and when it makes sense to migrate from one to another.

## 🧪 Methodology

- **Data**: synthetic dataset (varied volumes) + public dataset
- **Transformations tested**: filter, groupby, join, window, sort, string ops, combined pipeline
- **Execution**: N runs per combination (tool × transformation × volume), cold and warm start
- **Metrics collected**: execution time and memory usage
- **Environment**: isolated Docker containers for each tool (ensures reproducibility)

## 📁 Repository structure

```
├── data/                  # synthetic data generation and public dataset download
├── scripts/
│   ├── pandas/            # transformations implemented in Pandas
│   ├── polars/            # transformations implemented in Polars
│   └── spark/             # transformations implemented in Spark
├── benchmark/
│   ├── runner.py          # orchestrates the runs and collects metrics
│   └── results/           # result CSVs: {action}_{tool}_{volume}.csv
├── analysis/
│   └── results_analysis.ipynb   # comparative analysis and charts
├── requirements.txt
└── docker-compose.yml
```

## 🚀 How to run

```bash
# start the environments
docker-compose up -d

# run the full benchmark
python benchmark/runner.py

# view the analysis
jupyter notebook analysis/results_analysis.ipynb
```

## 📊 Results

_(to be filled in as the benchmarks are run)_

| Transformation | Volume | Pandas | Polars | Spark |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## 🔍 Main conclusions

_(to be filled in at the end of the project)_

## 🛠️ Stack

Python · Pandas · Polars · PySpark · Docker · Jupyter

## 📝 Context

Personal study and portfolio project.
