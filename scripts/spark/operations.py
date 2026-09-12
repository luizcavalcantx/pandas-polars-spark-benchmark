"""
Benchmark operations implemented in PySpark (local mode, a single JVM
on the machine). Since Spark is "lazy" (transformations only actually
execute when an action is called), count_rows() calls .count() -- and it's
this, inside the harness's timed block, that forces the real execution of the
operation. Without it, we would only be measuring the time to build the
execution plan, not to run it.

Run standalone (one operation, once):
    python scripts/spark/operations.py \
        --operation filter \
        --dataset data/synthetic/sales_1m.parquet \
        --dim-customers data/synthetic/dim_customers.parquet \
        --dim-products data/synthetic/dim_products.parquet \
        --repeats 3 --warmup 1
"""
import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.harness import run_cli

_spark = None


def get_spark():
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder.appName("benchmark")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")  # default (200) is excessive for running locally
            .config("spark.ui.showConsoleProgress", "false")  # avoids polluting stdout (runner.py reads the last line)
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def load_fact(path):
    return get_spark().read.parquet(path)


def load_dims(customers_path, products_path):
    spark = get_spark()
    return spark.read.parquet(customers_path), spark.read.parquet(products_path)


def count_rows(df):
    return df.count()


def op_filter(df, customers, products):
    return df.filter((F.col("status") == "completed") & (F.col("total_amount") > 100))


def op_groupby(df, customers, products):
    return df.groupBy("region", "payment_method").agg(
            F.sum("total_amount").alias("total"),
            F.avg("total_amount").alias("avg"),
            F.count("*").alias("n"),
    )


def op_join(df, customers, products):
    return (
        df
        .join(customers.select("customer_id", "segment"), "customer_id", "inner")
        .join(products.select("product_id", "category"), "product_id", "inner")
    )


def op_window(df, customers, products):
    w_region = Window.partitionBy("region").orderBy(F.col("total_amount").desc())
    result = df.withColumn("rank_in_region", F.dense_rank().over(w_region))

    w_customer = (
        Window.partitionBy("customer_id")
        .orderBy("order_date")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    result = result.withColumn("customer_cum_spend", F.sum("total_amount").over(w_customer))
    return result


def op_sort(df, customers, products):
    return df.orderBy(F.col("total_amount").desc(), F.col("order_date").asc())


def op_string_ops(df, customers, products):
    return (
        df.withColumn("email_domain", F.split(F.col("customer_email"), "@").getItem(1))
        .withColumn("customer_name_upper", F.upper(F.col("customer_name")))
        .withColumn("name_contains_silva", F.lower(F.col("customer_name")).contains("silva"))
    )


def op_pipeline(df, customers, products):
    filtered = df.filter(F.col("status") == "completed")
    merged = filtered.join(customers.select("customer_id", "segment"), "customer_id", "inner")
    merged = merged.join(products.select("product_id", "category"), "product_id", "inner")
    grouped = merged.groupBy("category", "segment").agg(
        F.sum("total_amount").alias("total"),
        F.avg("total_amount").alias("avg"),
        F.count("*").alias("n"),
    )
    return grouped.orderBy(F.col("total").desc())


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
    run_cli("spark", OPERATIONS, load_fact, load_dims, count_rows)
