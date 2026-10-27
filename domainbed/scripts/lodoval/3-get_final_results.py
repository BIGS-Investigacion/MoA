import argparse
import json
import re
from pathlib import Path

import pandas as pd

SCALAR_METRICS = ["acc", "bacc", "f1", "f1_pos0", "f1_pos1", "auroc"]
EXTRA_METRICS = ["loss", "macro_sensitivity", "macro_specificity"]
PER_CLASS_METRICS = ["sensitivity", "specificity"]  # en el log son dicts {clase: valor}
MINIMIZE = {"loss"}  # métricas de validación que se minimizan


def _flatten_per_class(rec):
    """env0_in_sensitivity = {"0": a, "1": b} -> env0_in_sensitivity_c0, ..._c1.

    Acepta dict (claves str o int tras el JSON) o lista.
    """
    for key in list(rec):
        if key.endswith(tuple(f"_{m}" for m in PER_CLASS_METRICS)) \
                and isinstance(rec[key], (dict, list)):
            value = rec.pop(key)
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for cls, v in items:
                rec[f"{key}_c{cls}"] = v
    return rec


def get_data(data_dir):
    """Lee TODOS los steps de cada results.jsonl."""
    rows = []
    for path in Path(data_dir).rglob("results.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = _flatten_per_class(json.loads(line))
                rec["log_file"] = str(path)
                rec["log_idx"] = i
                rows.append(rec)

    if not rows:
        raise ValueError(f"No hay results.jsonl con datos en {data_dir}")

    df = pd.json_normalize(rows, sep=".")

    # Sin --lodoval, real_test_envs tiene un solo entorno
    if (df["args.real_test_envs"].apply(len) != 1).any():
        raise ValueError("Hay logs con más de un test env (¿son de lodoval?).")

    df["test_env"] = df["args.real_test_envs"].str[0]
    df["model"] = df["hparams.model"]
    return df


def add_val_metric(df, metric):
    """val_metric = media de env{j}_out_{metric} sobre los dominios j != test_env."""
    df = df.copy()
    df["val_metric"] = float("nan")
    for te, idx in df.groupby("test_env").groups.items():
        cols = [c for c in df.columns
                if (m := re.fullmatch(rf"env(\d+)_out_{metric}", c))
                and int(m.group(1)) != te]
        if not cols:
            raise ValueError(
                f"No hay columnas env*_out_{metric} para test_env={te}. "
                "¿Se entrenó con rich_eval (--lodoval o --extra_metrics) "
                "y holdout_fraction > 0?")
        df.loc[idx, "val_metric"] = df.loc[idx, cols].mean(axis=1)
    return df


def select_best_step(df, metric):
    """Una fila por (results.jsonl, test_env): el step con mejor validación.

    Un mismo results.jsonl puede contener varios test_env (main.py los recorre
    en un solo proceso), y el step se reinicia en cada uno. Por eso se agrupa
    por ambos. En empate se queda el step más temprano.
    """
    df = add_val_metric(df, metric)
    df = df.dropna(subset=["val_metric"]).sort_values(
        ["log_file", "test_env", "step"])
    g = df.groupby(["log_file", "test_env"])["val_metric"]
    idx = g.idxmin() if metric in MINIMIZE else g.idxmax()
    return df.loc[idx].reset_index(drop=True)


def add_test_metrics(df, metrics):
    """Copia env{te}_in_{m} -> test_{m}; incluye sensibilidad/especificidad por clase."""
    classes = sorted({
        int(m.group(1)) for c in df.columns
        if (m := re.search(r"_(?:sensitivity|specificity)_c(\d+)$", c))
    })
    names = list(metrics) + [f"{m}_c{c}" for m in PER_CLASS_METRICS for c in classes]

    for n in names:
        df[f"test_{n}"] = float("nan")
    for te, idx in df.groupby("test_env").groups.items():
        for n in names:
            src = f"env{te}_in_{n}"
            if src in df.columns:
                df.loc[idx, f"test_{n}"] = df.loc[idx, src]
    return df, [f"test_{n}" for n in names]


def print_table(df, metrics):
    df, test_cols = add_test_metrics(df, metrics)

    index = ["dataset", "trial_seed", "test_env", "model", "algorithm"]
    cols = ["args.hparams_seed", "step", "val_metric"] + test_cols

    table = (
        df[["args.dataset", "args.trial_seed", "test_env", "model",
            "args.algorithm"] + cols]
        .rename(columns={"args.dataset": "dataset",
                         "args.trial_seed": "trial_seed",
                         "args.algorithm": "algorithm",
                         "args.hparams_seed": "hparams_seed"})
        .sort_values(index)
        .set_index(index)
    )

    dups = table.index.duplicated(keep=False)
    if dups.any():
        print(f"AVISO: {dups.sum()} filas con índice repetido:")
        print(table[dups])

    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(table)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="FINAL_RESULTS_v")
    parser.add_argument("--metrics", nargs="+",
                        default=SCALAR_METRICS + EXTRA_METRICS,
                        help="Métricas de test a reportar")
    parser.add_argument("--select", default="bacc",
                        choices=SCALAR_METRICS + EXTRA_METRICS,
                        help="Métrica de validación para elegir el step "
                             "(loss se minimiza, el resto se maximiza)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = select_best_step(get_data(args.data_dir), args.select)
    table = print_table(df, args.metrics)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.reset_index().to_csv(args.out, index=False)
        print(f"CSV guardado en {args.out}")


if __name__ == "__main__":
    main()