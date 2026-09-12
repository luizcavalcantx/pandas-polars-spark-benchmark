"""
Downloads and validates the public NYC Yellow Taxi Trip Records dataset.

The TLC (NYC Taxi & Limousine Commission) already provides the data in
Parquet, so no conversion is needed -- just download + validation.
Official page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Each monthly file has between ~2.5M and ~3.5M trips. To assemble a volume
close to 50M+ rows for the benchmark, download several months and concatenate
them (the script already does this by passing --months with a list of "YYYY-MM").

Usage:
    # one month (fast, to test the pipeline)
    python download_public_dataset.py --months 2024-01 --out-dir data/public/

    # several months concatenated into a single parquet (~ tens of millions of rows)
    python download_public_dataset.py \
        --months 2023-01 2023-02 2023-03 2023-04 2023-05 2023-06 \
                 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 \
                 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06 \
        --out-dir data/public/ --combined-out data/public/nyc_taxi_combined.parquet
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

import pyarrow.parquet as pq
import pyarrow as pa

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"

# Minimum expected columns (the TLC schema has changed over the years,
# so we only validate the stable subset used in the benchmark)
EXPECTED_COLUMNS = {
    "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "PULocationID", "DOLocationID", "payment_type",
    "fare_amount", "tip_amount", "total_amount",
}


def download_month(month: str, out_dir: Path) -> Path:
    url = BASE_URL.format(month=month)
    dest = out_dir / f"yellow_tripdata_{month}.parquet"
    if dest.exists():
        print(f"  {dest.name} already exists, skipping download")
        return dest
    print(f"  downloading {url}")
    try:
        urlretrieve(url, dest)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    return dest


def validate_file(path: Path) -> dict:
    table = pq.read_table(path)
    cols = set(table.column_names)
    missing = EXPECTED_COLUMNS - cols
    df_sample = table.slice(0, 0).to_pandas()  # only to get dtypes, without materializing everything

    report = {
        "file": path.name,
        "rows": table.num_rows,
        "columns": len(table.column_names),
        "missing_expected_columns": sorted(missing),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
    }

    if not missing:
        pickup = table.column("tpep_pickup_datetime")
        n_nulls_pickup = pickup.null_count
        dist = table.column("trip_distance").to_numpy(zero_copy_only=False)
        report["null_pickup_datetime"] = int(n_nulls_pickup)
        report["negative_or_zero_distance_pct"] = round(float((dist <= 0).mean()) * 100, 2)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", required=True, help="List of months in YYYY-MM format")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory where downloaded files are saved")
    parser.add_argument("--combined-out", type=str, default=None,
                         help="If provided, concatenates all months into this .parquet file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []
    print(f"Downloading {len(args.months)} month(s)...")
    for month in args.months:
        path = download_month(month, out_dir)
        downloaded_paths.append(path)

    print("\nValidating files:")
    reports = []
    for path in downloaded_paths:
        report = validate_file(path)
        reports.append(report)
        print(f"  {report}")

    total_rows = sum(r["rows"] for r in reports)
    print(f"\nTotal: {total_rows:,} rows in {len(reports)} file(s)")

    if args.combined_out:
        print(f"\nConcatenating into {args.combined_out} ...")
        tables = [pq.read_table(p) for p in downloaded_paths]
        # normalize to the schema of the first file (columns can vary
        # slightly between years, e.g. presence of 'airport_fee')
        common_cols = set(tables[0].column_names)
        for t in tables[1:]:
            common_cols &= set(t.column_names)
        common_cols = list(common_cols)
        tables = [t.select(common_cols) for t in tables]
        combined = pa.concat_tables(tables, promote_options="default")
        combined_path = Path(args.combined_out)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined, combined_path, compression="zstd")
        print(f"Combined: {combined.num_rows:,} rows -> {combined_path}")


if __name__ == "__main__":
    main()
