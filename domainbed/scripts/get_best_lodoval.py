import argparse
import glob
import json
import os
import re

import pandas as pd


def is_minimized(metric, mode="auto"):
    """
    Decide si la métrica de selección se minimiza (pérdida) o se maximiza
    (accuracy, etc.).

    mode="auto": se minimiza si el nombre de la métrica contiene "loss".
    mode="min" / "max": fuerza la dirección.
    """
    if mode == "min":
        return True
    if mode == "max":
        return False
    return "loss" in metric.lower()


def pick_best_idx(grouped_series, minimize):
    """idxmin si se minimiza, idxmax si se maximiza."""
    return grouped_series.idxmin() if minimize else grouped_series.idxmax()


def load_records(input_paths):
    """Carga todos los registros jsonl de los paths/globs dados."""
    records = []
    for pattern in input_paths:
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"[WARN] No se pudo parsear una línea en {path}, se omite.")
                        continue
                    r["_source_file"] = path
                    records.append(r)
    return records


def get_env_pair(args):
    """
    Devuelve el par (a, b) de dominios excluidos del entrenamiento en este run.

    En el framework nuevo, args["test_envs"] es la lista de TODOS los pares
    del sweep (p. ej. [[0,1],[0,2],...]) y es idéntica en todos los runs; el
    par real de cada run está en args["real_test_envs"].
    Como fallback se acepta el formato antiguo (test_envs = [a, b]).
    """
    pair = args.get("real_test_envs")
    if pair is None:
        pair = args.get("test_envs", [])
        # Formato nuevo sin real_test_envs: lista de pares -> no se puede saber
        if pair and isinstance(pair[0], (list, tuple)):
            return None
    if len(pair) != 2:
        return None
    return int(pair[0]), int(pair[1])


def get_model_name(r):
    """El nombre del modelo ahora vive en hparams["model"] (args no lo trae)."""
    args = r.get("args", {})
    return args.get("model") or r.get("hparams", {}).get("model")


def build_lodo_dataframe(records, metric="acc"):
    """
    Convierte los registros leave-two-out en observaciones del protocolo LODO,
    para TODOS los posibles dominios de test (en vez de uno fijo).

    Cada registro tiene real_test_envs=[a, b]: dos dominios excluidos del
    entrenamiento. Ese único registro puede jugar dos roles distintos:

        rol 1: val=a, test=b
        rol 2: val=b, test=a

    Por eso, cada línea del jsonl genera DOS filas en el DataFrame (una por
    cada elección de dominio de test), reutilizando las mismas métricas
    env{a}_in_{metric} / env{b}_in_{metric} que ya están en el registro.

    trial_seed SÍ define el experimento (se mantiene como eje independiente).
    seed NO define el experimento (es metadata, no se usa para agrupar).
    """
    rows = []
    val_col = f"val_{metric}"
    test_col = f"test_{metric}"

    for r in records:
        args = r.get("args", {})
        pair = get_env_pair(args)

        if pair is None:
            continue

        a, b = pair
        for val_env, test_env in ((a, b), (b, a)):
            val_key = f"env{val_env}_in_{metric}"
            test_key = f"env{test_env}_in_{metric}"

            if val_key not in r or test_key not in r:
                continue

            rows.append({
                "dataset": args.get("dataset"),
                "algorithm": args.get("algorithm"),
                "model": get_model_name(r),

                # seed NO define el experimento.
                "seed": args.get("seed"),

                # trial_seed SÍ define el experimento: cada trial_seed se
                # trata como un experimento independiente en todo el pipeline.
                "trial_seed": args.get("trial_seed"),

                "hparams_seed": args.get(
                    "hparams_seed", args.get("hparam_seed")
                ),
                "step": r.get("step"),

                "val_env": val_env,
                "test_env": test_env,

                val_col: r[val_key],
                test_col: r[test_key],

                "output_dir": args.get("output_dir"),
            })

    df = pd.DataFrame(rows)

    # Las pérdidas pueden ser NaN/inf si el entrenamiento diverge. Se
    # descartan esas filas para que idxmin/idxmax no fallen ni seleccionen
    # valores inválidos.
    if not df.empty:
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.dropna(subset=[val_col, test_col]).copy()
        df[val_col] = df[val_col].astype(float)
        df[test_col] = df[test_col].astype(float)

    return df


def select_best_step_per_run(df, metric="acc", minimize=False):
    """
    Para cada run individual (dataset, algorithm, model, trial_seed,
    hparams_seed, val_env, test_env), se queda con la época que alcanza el
    MEJOR valor de la métrica de validación (máximo si es accuracy, mínimo si
    es pérdida; selección tipo "oracle early-stopping" dentro del propio run).

    Esto colapsa el eje "step"/época, dejando un único punto (val, test) por
    cada combinación de run + hparams_seed + val_env.
    """
    val_col = f"val_{metric}"

    group_cols = [
        "dataset", "algorithm", "model",
        "trial_seed", "hparams_seed",
        "val_env", "test_env",
    ]

    idx = pick_best_idx(df.groupby(group_cols)[val_col], minimize)
    best_step = df.loc[idx].reset_index(drop=True)
    best_step = best_step.rename(columns={"step": "best_step"})
    return best_step


def compute_hparam_selection(best_step_df, metric="acc", minimize=False):
    """
    Selecciona, POR SEPARADO para cada (dataset, algorithm, model,
    trial_seed, test_env), el hparams_seed ganador.

    El score de cada hparams_seed es el promedio (sobre los val_env
    disponibles para ese test_env) del mejor valor de validación alcanzado
    en cualquiera de las épocas (ver select_best_step_per_run). El ganador es
    el de mayor score si se maximiza, o el de menor score si se minimiza
    (pérdida).

    Solo se comparan hparams_seed que tengan resultados para el conjunto
    COMPLETO de val_envs esperado para ese (trial_seed, test_env); si un
    hparams_seed tiene menos val_envs que otros, se descarta para mantener
    la comparación justa.
    """
    val_col = f"val_{metric}"
    mean_val_col = f"mean_{val_col}"
    std_val_col = f"std_{val_col}"

    config_cols = [
        "dataset", "algorithm", "model",
        "trial_seed", "test_env", "hparams_seed",
    ]
    group_keys = ["dataset", "algorithm", "model", "trial_seed", "test_env"]

    val_summary = (
        best_step_df.groupby(config_cols)
        .agg(**{
            mean_val_col: (val_col, "mean"),
            std_val_col: (val_col, "std"),
            "n_val_envs": ("val_env", "nunique"),
            "val_envs_used": ("val_env", lambda s: tuple(sorted(set(s)))),
        })
        .reset_index()
    )

    # Conjunto de val_envs "esperado" para cada (trial_seed, test_env): la
    # unión de val_envs vista entre todos los hparams_seed candidatos.
    expected = (
        val_summary
        .groupby(group_keys)["val_envs_used"]
        .apply(lambda s: tuple(sorted(set().union(*s))))
        .rename("expected_val_envs")
        .reset_index()
    )
    val_summary = val_summary.merge(expected, on=group_keys, how="left")

    val_summary = val_summary[
        val_summary["val_envs_used"] == val_summary["expected_val_envs"]
    ].drop(columns="expected_val_envs").copy()

    idx = pick_best_idx(val_summary.groupby(group_keys)[mean_val_col], minimize)
    best = val_summary.loc[idx].reset_index(drop=True)
    return best, val_summary


def compute_test_results(best_step_df, best_hparams, metric="acc"):
    """
    Recupera, para cada (dataset, algorithm, model, trial_seed, test_env),
    los resultados de test correspondientes al hparams_seed ganador,
    promediando sobre los val_env que participaron en esa selección.

    Cada trial_seed queda completamente disgregado: no se mezcla con otros
    trial_seed en ningún punto del pipeline.
    """
    val_col = f"val_{metric}"
    test_col = f"test_{metric}"
    mean_val_col = f"mean_{val_col}"
    mean_test_col = f"mean_{test_col}"
    std_test_col = f"std_{test_col}"

    keys = ["dataset", "algorithm", "model", "trial_seed", "test_env", "hparams_seed"]
    group_keys = ["dataset", "algorithm", "model", "trial_seed", "test_env"]

    selected = best_step_df.merge(
        best_hparams[keys],
        on=keys,
        how="inner",
    )

    test_summary = (
        selected
        .groupby(group_keys)
        .agg(**{
            mean_test_col: (test_col, "mean"),
            std_test_col: (test_col, "std"),
            "n_test_runs": (test_col, "count"),
            "n_val_envs": ("val_env", "nunique"),
            "val_envs_used": ("val_env", lambda s: tuple(sorted(set(s)))),
            "best_steps_used": ("best_step", lambda s: tuple(sorted(set(s)))),
            "hparams_seed": ("hparams_seed", "first"),
        })
        .reset_index()
    )

    test_summary = test_summary.merge(
        best_hparams[group_keys + [mean_val_col]],
        on=group_keys,
        how="left",
    )

    return test_summary


def _safe_component(value):
    """Convierte un valor en un nombre de carpeta seguro (sin separadores de ruta)."""
    s = str(value)
    s = re.sub(r"[\\/]+", "_", s)
    s = s.strip().strip(".")
    return s or "unknown"


def save_results_by_group(test_summary, out_dir):
    """
    Guarda test_summary en out_dir/dataset/algorithm/model/best.csv, un
    archivo por cada combinación (dataset, algorithm, model). Crea las
    carpetas automáticamente si no existen.
    """
    saved = []
    for (dataset, algorithm, model), sub in test_summary.groupby(
        ["dataset", "algorithm", "model"], dropna=False
    ):
        folder = os.path.join(
            out_dir,
            _safe_component(dataset),
            _safe_component(algorithm),
            _safe_component(model),
        )
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "best.csv")
        sub.to_csv(path, index=False)
        saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Archivo(s) jsonl o patrón(es) glob"
    )

    parser.add_argument(
        "--metric",
        default="acc",
        help=(
            "Métrica a usar como criterio de selección y de reporte "
            "(default: acc). Ej: acc, loss. Las métricas cuyo nombre "
            "contiene 'loss' se minimizan automáticamente."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "max", "min"],
        default="auto",
        help=(
            "Dirección de la optimización. 'auto' (default) minimiza si la "
            "métrica contiene 'loss' y maximiza en caso contrario; 'max' y "
            "'min' fuerzan la dirección."
        ),
    )

    parser.add_argument(
        "--test-envs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Restringe la salida a estos dominios de test. "
            "Por default se calculan TODOS los dominios de test observados."
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Carpeta de salida (opcional). Los resultados se guardan en "
            "<out>/<dataset>/<algorithm>/<model>/best.csv; las carpetas se "
            "crean automáticamente si no existen."
        ),
    )

    args = parser.parse_args()

    metric = args.metric
    minimize = is_minimized(metric, args.mode)
    direction = "mínimo" if minimize else "máximo"

    val_col = f"val_{metric}"
    test_col = f"test_{metric}"
    mean_val_col = f"mean_{val_col}"
    std_val_col = f"std_{val_col}"
    mean_test_col = f"mean_{test_col}"
    std_test_col = f"std_{test_col}"

    print(f"[INFO] Métrica: {metric} | criterio de selección: {direction}")

    # ---------------------------------------------------------
    # 1. Cargar registros
    # ---------------------------------------------------------

    records = load_records(args.inputs)

    if not records:
        print("No se encontraron registros.")
        return

    # ---------------------------------------------------------
    # 2. Convertir leave-two-out -> protocolo LODO, para TODOS los
    #    dominios de test posibles
    # ---------------------------------------------------------

    df = build_lodo_dataframe(records, metric=metric)

    if df.empty:
        print(
            "No se encontraron registros válidos para el protocolo LODO."
        )
        return

    if args.test_envs is not None:
        df = df[df["test_env"].isin(args.test_envs)].copy()
        if df.empty:
            print("No hay datos para los --test-envs solicitados.")
            return

    # ---------------------------------------------------------
    # 3. Por cada run, quedarse con la época que da el mejor valor de la
    #    métrica de validación (colapsa el eje step/época)
    # ---------------------------------------------------------

    best_step_df = select_best_step_per_run(df, metric=metric, minimize=minimize)

    # ---------------------------------------------------------
    # 4. Seleccionar hparams_seed usando SOLO validación, por separado
    #    para cada trial_seed y cada test_env
    # ---------------------------------------------------------

    best_hparams, val_summary = compute_hparam_selection(
        best_step_df, metric=metric, minimize=minimize
    )

    print(
        f"\n=== Mejor hparams_seed por trial_seed y dominio de test "
        f"(seleccionado SOLO con validación, {direction} por época) ===\n"
    )

    cols = [
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
        "hparams_seed",
        mean_val_col,
        std_val_col,
        "n_val_envs",
        "val_envs_used",
    ]

    print(
        best_hparams[cols]
        .sort_values([
            "dataset",
            "algorithm",
            "model",
            "trial_seed",
            "test_env",
        ])
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # 5. Evaluar esa configuración en el/los dominio(s) de test,
    #    por separado para cada trial_seed
    # ---------------------------------------------------------

    test_summary = compute_test_results(
        best_step_df,
        best_hparams,
        metric=metric,
    )

    print(
        "\n=== Resultado final por trial_seed y dominio de test "
        "(cada trial_seed es un experimento independiente) ===\n"
    )

    cols_test = [
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
        "hparams_seed",
        mean_val_col,
        mean_test_col,
        std_test_col,
        "n_test_runs",
        "n_val_envs",
        "val_envs_used",
        "best_steps_used",
    ]

    test_summary_sorted = test_summary[cols_test].sort_values([
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
    ])

    print(test_summary_sorted.to_string(index=False))

    # ---------------------------------------------------------
    # 5b. Vista pivotada: dominios de test como columnas, para
    #     comparar rápido entre trial_seed
    # ---------------------------------------------------------

    pivot = test_summary.pivot_table(
        index=["dataset", "algorithm", "model", "trial_seed"],
        columns="test_env",
        values=mean_test_col,
    )
    pivot.columns = [f"test_env_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    print(
        "\n=== Vista pivotada: mean_test_{metric} por dominio de test "
        "(columnas) y trial_seed (filas) ===\n".format(metric=metric)
    )
    print(pivot.to_string(index=False))

    # ---------------------------------------------------------
    # 6. Guardar: out/dataset/algorithm/model/best.csv
    # ---------------------------------------------------------

    if args.out:
        saved = save_results_by_group(test_summary_sorted, args.out)
        print(f"\nResultados guardados en {len(saved)} archivo(s):")
        for p in saved:
            print(f"  - {p}")


if __name__ == "__main__":
    main()