import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.common.harness import run_cli

def load_fact(path):
    return pd.read_parquet(path)

def load_dims(custumer_path, products_path):
    return pd.read_parquet(custumer_path), pd.read_parquet(products_path)

def count_rows(df):
    return len(df)

def op_filter(df, custumer, product):
    return df[(df['status'] == 'completed') & (df['total_amount'] > 100)]

def op_groupby(df, custumers, products):
    return (
        df.groupby(['region', 'payment_method'])
        .agg(total=('total_amount', 'sum'), avg=('total_amount', 'mean'), n=('total_amount','size'))
        .reset_index()
    )

def op_join (df, customers, products):
    merged = df.merge(customers[['customer_id','segment']], on='customer_id', how='inner')
    merged = merged.merge(products[['product_id','category']], on='product_id', how='inner')
    return merged

def op_window(df, customers, products):
    result = df.copy()
    # rank of the order value within each region
    result["rank_in_region"] = result.groupby("region")["total_amount"].rank(method="dense", ascending=False)
    # cumulative spend per customer, ordered over time
    result = result.sort_values(["customer_id", "order_date"])
    result["customer_cum_spend"] = result.groupby("customer_id")["total_amount"].cumsum()
    return result

def op_sort(df, customers, products):
    return df.sort_values(["total_amount", "order_date"], ascending=[False, True])

def op_string_ops(df, customers, products):
    result = df.copy()
    result["email_domain"] = result["customer_email"].str.split("@").str[-1]
    result["customer_name_upper"] = result["customer_name"].str.upper()
    result["name_contains_silva"] = result["customer_name"].str.contains("silva", case=False, na=False)
    return result

def op_pipeline(df, customers, products):
    filtered = df[df["status"] == "completed"]
    merged = filtered.merge(customers[["customer_id", "segment"]], on="customer_id", how="inner")
    merged = merged.merge(products[["product_id", "category"]], on="product_id", how="inner")
    grouped = (
        merged.groupby(["category", "segment"])
        .agg(total=("total_amount", "sum"), avg=("total_amount", "mean"), n=("total_amount", "size"))
        .reset_index()
    )
    return grouped.sort_values("total", ascending=False)

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
    run_cli("pandas", OPERATIONS, load_fact, load_dims, count_rows)
