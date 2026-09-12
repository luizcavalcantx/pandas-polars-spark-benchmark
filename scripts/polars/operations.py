"""
Benchmark operations implemented in Polars (eager API -- pl.read_parquet
loads everything into memory at once, to keep it comparable to Pandas; a
separate benchmark using pl.scan_parquet/LazyFrame would be interesting
as a future study, but it's not the focus of this phase).

Run standalone (one operation, once):
    python scripts/polars/operations.py \
        --operation filter \
        --dataset data/synthetic/sales_1m.parquet \
        --dim-customers data/synthetic/dim_customers.parquet \
        --dim-products data/synthetic/dim_products.parquet \
        --repeats 3 --warmup 1
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.harness import run_cli


def load_fact(path):
    return pl.read_parquet(path)


def load_dims(customers_path, products_path):
    return pl.read_parquet(customers_path), pl.read_parquet(products_path)


def count_rows(df):
    return df.height


def op_filter(df, customers, products):
    return df.filter((pl.col("status") == "completed") & (pl.col("total_amount") > 100))


def op_groupby(df, customers, products):
    return df.group_by(["region", "payment_method"]).agg(
        pl.col("total_amount").sum().alias("total"),
        pl.col("total_amount").mean().alias("avg"),
        pl.len().alias("n"),
    )


def op_join(df, customers, products):
    merged = df.join(customers.select(["customer_id", "segment"]), on="customer_id", how="inner")
    merged = merged.join(products.select(["product_id", "category"]), on="product_id", how="inner")
    return merged


def op_window(df, customers, products):
    result = df.with_columns(
        pl.col("total_amount").rank(method="dense", descending=True).over("region").alias("rank_in_region")
    )
    result = result.sort(["customer_id", "order_date"])
    result = result.with_columns(
        pl.col("total_amount").cum_sum().over("customer_id").alias("customer_cum_spend")
    )
    return result


def op_sort(df, customers, products):
    return df.sort(["total_amount", "order_date"], descending=[True, False])


def op_string_ops(df, customers, products):
    return df.with_columns(
        pl.col("customer_email").str.split("@").list.last().alias("email_domain"),
        pl.col("customer_name").str.to_uppercase().alias("customer_name_upper"),
        pl.col("customer_name").str.contains("(?i)silva").alias("name_contains_silva"),
    )


def op_pipeline(df, customers, products):
    filtered = df.filter(pl.col("status") == "completed")
    merged = filtered.join(customers.select(["customer_id", "segment"]), on="customer_id", how="inner")
    merged = merged.join(products.select(["product_id", "category"]), on="product_id", how="inner")
    grouped = merged.group_by(["category", "segment"]).agg(
        pl.col("total_amount").sum().alias("total"),
        pl.col("total_amount").mean().alias("avg"),
        pl.len().alias("n"),
    )
    return grouped.sort("total", descending=True)


OPERATIONS = {
    "filter": op_filter,
    "groupby": op_groupby,
    "join": op_join,
    "window": op_window,
    "sort": op_sort,
    "string_ops": op_string_ops,
    "pipeline": op_pipeline,
}


if __name__ == "__main__":
    run_cli("polars", OPERATIONS, load_fact, load_dims, count_rows)
