"""
Harness compartilhado pelos 3 scripts de operacoes (scripts/pandas,
scripts/polars, scripts/spark). Cada um desses scripts so precisa fornecer:

    OPERATIONS   dict: nome_da_operacao -> funcao(df, customers, products) -> resultado
    load_fact    funcao(path) -> dados carregados na estrutura da ferramenta
    load_dims    funcao(customers_path, products_path) -> (customers, products)
    count_rows   funcao(resultado) -> int (forca a materializacao em engines lazy, ex.: Spark)

O harness cuida do resto: le os argumentos da linha de comando, roda
warm-up (nao cronometrado), roda N repeticoes cronometradas e imprime o
resultado como UMA linha de JSON em stdout -- e assim que o
benchmark/runner.py le o resultado de cada processo filho.

Por que contar as linhas DENTRO do bloco cronometrado:
    Pandas e Polars (no modo eager usado aqui) ja materializam o resultado
    ao rodar a operacao. O Spark, por padrao, e "preguicoso" (lazy) -- as
    transformacoes so sao realmente executadas quando uma acao (como
    .count()) e chamada. Colocando count_rows() dentro do tempo medido,
    garantimos que TODAS as ferramentas sejam cronometradas pelo mesmo
    criterio: "tempo ate o resultado estar pronto de verdade".
"""
import argparse
import json
import time


def run_cli(tool_name, operations, load_fact, load_dims, count_rows):
    parser = argparse.ArgumentParser(description=f"Roda uma operacao do benchmark com {tool_name}")
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

    # warm-up: garante que imports tardios, JIT, cache de arquivo em disco
    # etc. nao distorcam a primeira medicao
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
