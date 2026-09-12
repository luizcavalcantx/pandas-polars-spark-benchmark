"""
Harness shared by the 3 operations scripts (scripts/pandas,
scripts/polars, scripts/spark). Each of these scripts only needs to provide:

    OPERATIONS   dict: operation_name -> function(df, customers, products) -> result
    load_fact    function(path) -> data loaded into the tool's structure
    load_dims    function(customers_path, products_path) -> (customers, products)
    count_rows   function(result) -> int (forces materialization in lazy engines, e.g. Spark)

The harness takes care of the rest: reads the command-line arguments, runs
warm-up (not timed), runs N timed repeats and prints the
result as ONE line of JSON to stdout -- this is how
benchmark/runner.py reads the result of each child process.

Why count the rows INSIDE the timed block:
    Pandas and Polars (in the eager mode used here) already materialize the result
    when running the operation. Spark, by default, is "lazy" -- the
    transformations are only actually executed when an action (like
    .count()) is called. By placing count_rows() inside the measured time, we
    ensure ALL tools are timed by the same
    criterion: "time until the result is actually ready".
"""
import argparse
import json
import time


def run_cli(tool_name, operations, load_fact, load_dims, count_rows):
    parser = argparse.ArgumentParser(description=f"Runs a benchmark operation with {tool_name}")
    parser.add_argument("--operation", required=True, choices=list(operations.keys()))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dim-customers", required=True)
    parser.add_argument("--dim-products", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    df = load_fact(args.dataset)
    customers, products = load_dims(args.dim_customers, args.dim_products)
    input_rows = count_rows(df)

    op_fn = operations[args.operation]

    # warm-up: ensures late imports, JIT, on-disk file cache
    # etc. don't skew the first measurement
    for _ in range(args.warmup):
        result = op_fn(df, customers, products)
        count_rows(result)

    times = []
    output_rows = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        result = op_fn(df, customers, products)
        output_rows = count_rows(result)
        times.append(time.perf_counter() - t0)

    print(json.dumps({
        "tool": tool_name,
        "operation": args.operation,
        "dataset": args.dataset,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "times_sec": times,
    }))
