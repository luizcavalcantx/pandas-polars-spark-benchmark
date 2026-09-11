"""
Verifica rapidamente os arquivos .parquet gerados: numero de linhas e
tamanho em disco de cada um, sem carregar os dados na memoria (le so o
metadata do Parquet).

Uso:
    python check_parquet.py
    python check_parquet.py --dir data/public
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="data/synthetic", help="Pasta com os .parquet a checar")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        print(f"Pasta nao encontrada: {folder.resolve()}")
        return

    files = sorted(folder.glob("*.parquet"))
    if not files:
        print(f"Nenhum .parquet encontrado em {folder.resolve()}")
        return

    print(f"Arquivos em {folder.resolve()}:\n")
    for f in files:
        pf = pq.ParquetFile(f)
        rows = pf.metadata.num_rows
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} {rows:>15,} linhas   {size_mb:>10,.1f} MB")


if __name__ == "__main__":
    main()