"""
Synthetic data generator for the Pandas vs Polars vs Spark benchmark.

Has TWO subcommands:

  dimensions  -> generates the dimension tables (dim_customers, dim_products),
                 used in the JOIN benchmark. Run it once; they are
                 reused by all fact table volumes.

  fact        -> generates the "sales_transactions" fact table in Parquet, in
                 chunks, supporting volumes from 1M to 200M+ rows without
                 blowing up RAM.

Schema (sales_transactions):
    order_id         int64    - primary key, sequential
    order_date       date32   - used in sort / window (partition by + order by)
    customer_id      int32    - FK to dim_customers, used in join and groupby
    product_id       int32    - FK to dim_products, used in join
    store_id         int16    - used in groupby
    region           string   - low cardinality, used in filter/groupby
    payment_method   string   - low cardinality, used in filter/groupby
    channel          string   - low cardinality, used in filter/groupby
    status           string   - low cardinality, used in filter (e.g. status == 'completed')
    quantity         int16
    unit_price       float32
    discount_pct     float32
    total_amount     float32  - quantity * unit_price * (1 - discount_pct)
    customer_name    string   - used in string ops (split, upper/lower, contains)
    customer_email   string   - used in string ops (extract domain, regex)

Usage:
    # 1) dimension tables (only once)
    python generate_synthetic_data.py dimensions --out-dir data/synthetic/

    # 2) fact table, one volume at a time
    python generate_synthetic_data.py fact --rows 1_000_000   --out data/synthetic/sales_1m.parquet
    python generate_synthetic_data.py fact --rows 10_000_000  --out data/synthetic/sales_10m.parquet
    python generate_synthetic_data.py fact --rows 50_000_000  --out data/synthetic/sales_50m.parquet  --chunk-size 5_000_000
    python generate_synthetic_data.py fact --rows 200_000_000 --out data/synthetic/sales_200m.parquet --chunk-size 5_000_000

Notes:
    - For 50M/200M rows, use --chunk-size to control the peak RAM
      (each chunk is materialized in memory and then discarded).
    - The final file is a SINGLE .parquet (multiple row groups), which
      makes it easier to compare read/scan performance between Pandas, Polars and Spark.
    - Use --seed for reproducibility (each chunk uses a derived seed,
      so the result doesn't change depending on the chosen --chunk-size).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Fixed domains (shared between dimensions and fact)
# ---------------------------------------------------------------------------
N_CUSTOMERS = 2_000_000
N_PRODUCTS = 5_000
N_STORES = 500

REGIONS = np.array(["North", "Northeast", "Midwest", "Southeast", "South"])
PAYMENT_METHODS = np.array(["credit_card", "debit_card", "pix", "boleto", "cash"])
CHANNELS = np.array(["online", "physical_store", "marketplace"])
STATUSES = np.array(["completed", "cancelled", "returned", "pending"])
STATUS_WEIGHTS = np.array([0.80, 0.08, 0.07, 0.05])
DISCOUNTS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30], dtype=np.float32)
DISCOUNT_WEIGHTS = np.array([0.50, 0.20, 0.15, 0.08, 0.05, 0.02])
EMAIL_DOMAINS = np.array(["gmail.com", "hotmail.com", "outlook.com", "yahoo.com.br"])

FIRST_NAMES = np.array([
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Hugo",
    "Isabela", "Joao", "Karina", "Lucas", "Mariana", "Nicolas", "Olivia", "Pedro",
    "Queila", "Rafael", "Sabrina", "Thiago", "Ursula", "Vinicius", "Wesley",
    "Ximena", "Yasmin", "Zeca", "Camila", "Diego", "Elaine", "Fernando",
])
LAST_NAMES = np.array([
    "Silva", "Souza", "Oliveira", "Santos", "Pereira", "Costa", "Rodrigues",
    "Almeida", "Nascimento", "Lima", "Araujo", "Fernandes", "Carvalho", "Gomes",
    "Martins", "Rocha", "Ribeiro", "Alves", "Monteiro", "Cardoso", "Teixeira",
    "Moreira", "Correia", "Cavalcanti", "Dias", "Castro", "Campos", "Barros",
])

COUNTRIES = np.array(["BR", "PT", "AR", "US", "CL"])
SEGMENTS = np.array(["consumer", "small_business", "enterprise"])
SEGMENT_WEIGHTS = np.array([0.70, 0.22, 0.08])
PRODUCT_CATEGORIES = np.array([
    "electronics", "fashion", "home_and_decor", "sports_and_leisure",
    "books", "beauty", "food", "toys", "computing", "pet_shop",
])
SUPPLIERS = np.array([f"supplier_{i:03d}" for i in range(1, 51)])

FACT_SCHEMA = pa.schema([
    ("order_id", pa.int64()),
    ("order_date", pa.date32()),
    ("customer_id", pa.int32()),
    ("product_id", pa.int32()),
    ("store_id", pa.int16()),
    ("region", pa.string()),
    ("payment_method", pa.string()),
    ("channel", pa.string()),
    ("status", pa.string()),
    ("quantity", pa.int16()),
    ("unit_price", pa.float32()),
    ("discount_pct", pa.float32()),
    ("total_amount", pa.float32()),
    ("customer_name", pa.string()),
    ("customer_email", pa.string()),
])

FACT_DATE_START = np.datetime64("2022-01-01", "D")
FACT_DATE_END = np.datetime64("2025-12-31", "D")
FACT_DATE_RANGE_DAYS = int((FACT_DATE_END - FACT_DATE_START).astype(int))

DIM_DATE_START = np.datetime64("2018-01-01", "D")
DIM_DATE_END = np.datetime64("2025-12-31", "D")
DIM_DATE_RANGE_DAYS = int((DIM_DATE_END - DIM_DATE_START).astype(int))


# ---------------------------------------------------------------------------
# Subcommand: dimensions
# ---------------------------------------------------------------------------
def build_customers(rng: np.random.Generator) -> pa.Table:
    customer_id = np.arange(1, N_CUSTOMERS + 1, dtype=np.int32)
    signup_offset = rng.integers(0, DIM_DATE_RANGE_DAYS, size=N_CUSTOMERS)
    signup_date = DIM_DATE_START + signup_offset.astype("timedelta64[D]")
    country = COUNTRIES[rng.integers(0, len(COUNTRIES), size=N_CUSTOMERS)]
    is_br = rng.random(N_CUSTOMERS) < 0.85
    country = np.where(is_br, "BR", country)
    segment = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_WEIGHTS)

    return pa.table({
        "customer_id": customer_id,
        "signup_date": pa.array(signup_date, type=pa.date32()),
        "country": country,
        "segment": segment,
    })


def build_products(rng: np.random.Generator) -> pa.Table:
    product_id = np.arange(1, N_PRODUCTS + 1, dtype=np.int32)
    category = PRODUCT_CATEGORIES[rng.integers(0, len(PRODUCT_CATEGORIES), size=N_PRODUCTS)]
    supplier = SUPPLIERS[rng.integers(0, len(SUPPLIERS), size=N_PRODUCTS)]
    base_price = rng.uniform(9.90, 899.90, size=N_PRODUCTS).astype(np.float32)
    product_name = np.array([f"{cat}_product_{pid:05d}" for cat, pid in zip(category, product_id)])

    return pa.table({
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "supplier": supplier,
        "base_price": base_price,
    })


def run_dimensions(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    customers = build_customers(rng)
    pq.write_table(customers, out_dir / "dim_customers.parquet", compression="zstd")
    print(f"dim_customers.parquet -> {customers.num_rows:,} rows")

    products = build_products(rng)
    pq.write_table(products, out_dir / "dim_products.parquet", compression="zstd")
    print(f"dim_products.parquet  -> {products.num_rows:,} rows")


# ---------------------------------------------------------------------------
# Subcommand: fact
# ---------------------------------------------------------------------------
def generate_fact_chunk(rng: np.random.Generator, start_id: int, size: int) -> pa.Table:
    order_id = np.arange(start_id, start_id + size, dtype=np.int64)

    offsets = rng.integers(0, FACT_DATE_RANGE_DAYS, size=size)
    order_date = FACT_DATE_START + offsets.astype("timedelta64[D]")

    customer_id = rng.integers(1, N_CUSTOMERS + 1, size=size, dtype=np.int32)
    product_id = rng.integers(1, N_PRODUCTS + 1, size=size, dtype=np.int32)
    store_id = rng.integers(1, N_STORES + 1, size=size).astype(np.int16)

    region = REGIONS[rng.integers(0, len(REGIONS), size=size)]
    payment_method = PAYMENT_METHODS[rng.integers(0, len(PAYMENT_METHODS), size=size)]
    channel = CHANNELS[rng.integers(0, len(CHANNELS), size=size)]
    status = rng.choice(STATUSES, size=size, p=STATUS_WEIGHTS)

    quantity = rng.integers(1, 11, size=size).astype(np.int16)
    unit_price = rng.uniform(9.90, 899.90, size=size).astype(np.float32)
    discount_pct = rng.choice(DISCOUNTS, size=size, p=DISCOUNT_WEIGHTS)
    total_amount = (quantity * unit_price * (1 - discount_pct)).astype(np.float32)

    first_idx = rng.integers(0, len(FIRST_NAMES), size=size)
    last_idx = rng.integers(0, len(LAST_NAMES), size=size)
    first = FIRST_NAMES[first_idx]
    last = LAST_NAMES[last_idx]
    customer_name = np.char.add(np.char.add(first, " "), last)

    email_local = np.char.add(np.char.lower(first), np.char.lower(last))
    suffix = rng.integers(1, 999, size=size).astype("U3")
    domains = EMAIL_DOMAINS[rng.integers(0, len(EMAIL_DOMAINS), size=size)]
    customer_email = np.char.add(np.char.add(np.char.add(email_local, suffix), "@"), domains)

    data = {
        "order_id": order_id,
        "order_date": order_date,
        "customer_id": customer_id,
        "product_id": product_id,
        "store_id": store_id,
        "region": region,
        "payment_method": payment_method,
        "channel": channel,
        "status": status,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "total_amount": total_amount,
        "customer_name": customer_name,
        "customer_email": customer_email,
    }
    return pa.Table.from_pydict(data, schema=FACT_SCHEMA)


def run_fact(args):
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = args.rows
    chunk_size = min(args.chunk_size, total)
    n_chunks = (total + chunk_size - 1) // chunk_size

    writer = None
    start_id = 1
    remaining = total
    t0 = time.time()

    print(f"Generating {total:,} rows in {n_chunks} chunk(s) of up to {chunk_size:,} -> {out_path}")

    for i in range(n_chunks):
        size = min(chunk_size, remaining)
        rng = np.random.default_rng(args.seed + i)
        table = generate_fact_chunk(rng, start_id, size)

        if writer is None:
            writer = pq.ParquetWriter(
                out_path, FACT_SCHEMA,
                compression=None if args.compression == "none" else args.compression,
            )
        writer.write_table(table)

        start_id += size
        remaining -= size
        elapsed = time.time() - t0
        done = total - remaining
        print(f"  chunk {i + 1}/{n_chunks}: {done:,}/{total:,} rows ({elapsed:.1f}s)")

    if writer is not None:
        writer.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done in {time.time() - t0:.1f}s. File: {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generates synthetic data (dimensions + fact) for the benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_dim = subparsers.add_parser("dimensions", help="Generates dim_customers.parquet and dim_products.parquet")
    p_dim.add_argument("--out-dir", type=str, required=True)
    p_dim.add_argument("--seed", type=int, default=7)
    p_dim.set_defaults(func=run_dimensions)

    p_fact = subparsers.add_parser("fact", help="Generates the sales_transactions.parquet fact table")
    p_fact.add_argument("--rows", type=int, required=True)
    p_fact.add_argument("--out", type=str, required=True)
    p_fact.add_argument("--chunk-size", type=int, default=2_000_000)
    p_fact.add_argument("--seed", type=int, default=42)
    p_fact.add_argument("--compression", type=str, default="zstd", choices=["snappy", "zstd", "gzip", "none"])
    p_fact.set_defaults(func=run_fact)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
