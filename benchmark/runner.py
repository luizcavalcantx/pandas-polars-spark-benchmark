"""
Orquestra a execucao do benchmark.

Para cada combinacao (tool, dataset, operacao), roda o script
scripts/<tool>/operations.py EM UM PROCESSO SEPARADO (subprocess) -- isso e
importante por dois motivos:
  1. O Spark sobe sua propria JVM; rodar tudo no mesmo processo Python
     misturaria o overhead/memoria da JVM com o das outras ferramentas.
  2. Isolamento de memoria: assim o pico de RAM medido para uma operacao
     nao inclui "sobras" (garbage nao coletado) de operacoes anteriores.

Mede tempo (retornado pelo proprio script filho, via harness.py) e pico de
memoria (RSS do processo + filhos, via psutil, amostrado em uma thread
enquanto o processo roda). Salva tudo, linha a linha, em
benchmark/results/results.csv.

Uso:
    python benchmark/runner.py
    python benchmark/runner.py --tools pandas polars --datasets 1m 10m
    python benchmark/runner.py --operations filter groupby --datasets 1m
"""
import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.common.config import (
    DATASETS, DIM_CUSTOMERS, DIM_PRODUCTS, OPERATIONS, TOOLS,
    REPEATS, WARMUP_RUNS, RESULTS_CSV,
)

SCRIPT_MAP = {
    "pandas": PROJECT_ROOT / "scripts/pandas/operations.py",
    "polars": PROJECT_ROOT / "scripts/polars/operations.py",
    "spark": PROJECT_ROOT / "scripts/spark/operations.py",
}


def _monitor_peak_memory(pid: int, stop_event: threading.Event, out: dict, interval: float = 0.05):
    """Amostra RSS (processo + filhos, ex.: JVM do Spark) ate stop_event ser sinalizado."""
    peak = 0
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        out["peak_bytes"] = 0
        return
    while not stop_event.is_set():
        try:
            procs = [proc] + proc.children(recursive=True)
            total = sum(p.memory_info().rss for p in procs if p.is_running())
            peak = max(peak, total)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(interval)
    out["peak_bytes"] = peak


def run_combo(tool: str, dataset_key: str, dataset_path: Path, operation: str):
    script = SCRIPT_MAP[tool]
    cmd = [
        sys.executable, str(script),
        "--operation", operation,
        "--dataset", str(dataset_path),
        "--dim-customers", str(DIM_CUSTOMERS),
        "--dim-products", str(DIM_PRODUCTS),
        "--repeats", str(REPEATS),
        "--warmup", str(WARMUP_RUNS),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop_event = threading.Event()
    mem_out = {}
    watcher = threading.Thread(target=_monitor_peak_memory, args=(proc.pid, stop_event, mem_out), daemon=True)
    watcher.start()

    stdout, stderr = proc.communicate()
    stop_event.set()
    watcher.join(timeout=2)

    if proc.returncode != 0:
        print(f"    [ERRO] {stderr.strip()[-800:]}")
        return None

    stdout_lines = [line for line in stdout.strip().splitlines() if line.strip()]
    if not stdout_lines:
        print(f"    [ERRO] processo nao imprimiu resultado. stderr: {stderr.strip()[-500:]}")
        return None

    payload = json.loads(stdout_lines[-1])
    payload["dataset_key"] = dataset_key
    payload["peak_memory_mb"] = round(mem_out.get("peak_bytes", 0) / (1024 * 1024), 1)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    return payload


def main():
    parser = argparse.ArgumentParser(description="Roda o benchmark Pandas vs Polars vs Spark")
    parser.add_argument("--tools", nargs="+", default=TOOLS, choices=TOOLS)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()), choices=list(DATASETS.keys()))
    parser.add_argument("--operations", nargs="+", default=OPERATIONS, choices=OPERATIONS)
    args = parser.parse_args()

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()

    fieldnames = [
        "timestamp", "tool", "dataset", "operation",
        "input_rows", "output_rows",
        "repeat_times_sec", "min_time_sec", "mean_time_sec", "peak_memory_mb",
    ]

    total = len(args.tools) * len(args.datasets) * len(args.operations)
    done = 0

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for dataset_key in args.datasets:
            dataset_path = DATASETS[dataset_key]
            if not dataset_path.exists():
                print(f"[AVISO] dataset '{dataset_key}' nao encontrado em {dataset_path}, pulando")
                continue

            for operation in args.operations:
                for tool in args.tools:
                    done += 1
                    print(f"[{done}/{total}] {tool} / {dataset_key} / {operation} ...")
                    result = run_combo(tool, dataset_key, dataset_path, operation)
                    if result is None:
                        continue

                    row = {
                        "timestamp": result["timestamp"],
                        "tool": result["tool"],
                        "dataset": result["dataset_key"],
                        "operation": result["operation"],
                        "input_rows": result["input_rows"],
                        "output_rows": result["output_rows"],
                        "repeat_times_sec": json.dumps(result["times_sec"]),
                        "min_time_sec": round(min(result["times_sec"]), 4),
                        "mean_time_sec": round(sum(result["times_sec"]) / len(result["times_sec"]), 4),
                        "peak_memory_mb": result["peak_memory_mb"],
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"    min={row['min_time_sec']}s  mean={row['mean_time_sec']}s  peak_mem={row['peak_memory_mb']}MB")

    print(f"\nResultados salvos em {RESULTS_CSV}")


if __name__ == "__main__":
    main()
