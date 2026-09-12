"""
Quickly checks the generated .parquet files: number of rows and
disk size of each one, without loading the data into memory (only reads
the Parquet metadata).

Usage:
    python check_parquet.py
    python check_parquet.py --dir data/public
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/synthetic", help="Folder with the .parquet files to check")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}")
        return

    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"No .parquet found in {folder.resolve()}")
        return

    print(f"Files in {folder.resolve()}:\n")
    for f in files:
        pf = pq.ParquetFile(f)
        rows = pf.metadata.num_rows
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {rows:>15,} rows   {size_mb:>10,.1f} MB")


if __name__ == "__main__":
    main()
