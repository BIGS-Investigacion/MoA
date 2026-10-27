import argparse
import glob
import json
import os
import re

import pandas as pd


def is_minimized(metric, mode="auto"):
    """
    Decide whether the selection metric is minimized (loss) or maximized
    (accuracy, etc.).

    mode="auto": minimized if the metric name contains "loss".
    mode="min" / "max": forces the direction.
    """
    if mode == "min":
        return True
    if mode == "max":
        return False
    return "loss" in metric.lower()


def pick_best_idx(grouped_series, minimize):
    """idxmin if minimizing, idxmax if maximizing."""
    return grouped_series.idxmin() if minimize else grouped_series.idxmax()


def load_records(input_paths):
    """Load all jsonl records from the given paths/globs."""
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
                        print(f"[WARN] Could not parse a line in {path}, skipping.")
                        continue
                    r["_source_file"] = path
                    records.append(r)
    return records


def get_env_pair(args):
    """
    Return the pair (a, b) of domains excluded from training in this run.

    In the new framework, args["test_envs"] is the list of ALL the pairs in
    the sweep (e.g. [[0,1],[0,2],...]) and is identical across all runs; the
    actual pair of each run is in args["real_test_envs"].
    As a fallback, the old format (test_envs = [a, b]) is accepted.
    """
    pair = args.get("real_test_envs")
    if pair is None:
        pair = args.get("test_envs", [])
        # New format without real_test_envs: list of pairs -> cannot be determined
        if pair and isinstance(pair[0], (list, tuple)):
            return None
    if len(pair) != 2:
        return None
    return int(pair[0]), int(pair[1])


def get_model_name(r):
    """The model name now lives in hparams["model"] (args does not carry it)."""
    args = r.get("args", {})
    return args.get("model") or r.get("hparams", {}).get("model")


def build_lodo_dataframe(records, metric="acc"):
    """
    Convert the leave-two-out records into LODO protocol observations,
    for ALL possible test domains (instead of a fixed one).

    Each record has real_test_envs=[a, b]: two domains excluded from
    training. That single record can play two different roles:

        role 1: val=a, test=b
        role 2: val=b, test=a

    Therefore, each line of the jsonl generates TWO rows in the DataFrame
    (one per choice of test domain), reusing the same metrics
    env{a}_in_{metric} / env{b}_in_{metric} that are already in the record.

    trial_seed DOES define the experiment (it is kept as an independent axis).
    seed does NOT define the experiment (it is metadata, not used for grouping).
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

                # seed does NOT define the experiment.
                "seed": args.get("seed"),

                # trial_seed DOES define the experiment: each trial_seed is
                # treated as an independent experiment throughout the pipeline.
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

    # Losses can be NaN/inf if training diverges. Those rows are dropped so
    # that idxmin/idxmax neither fail nor select invalid values.
    if not df.empty:
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.dropna(subset=[val_col, test_col]).copy()
        df[val_col] = df[val_col].astype(float)
        df[test_col] = df[test_col].astype(float)

    return df


def select_best_step_per_run(df, metric="acc", minimize=False):
    """
    For each individual run (dataset, algorithm, model, trial_seed,
    hparams_seed, val_env, test_env), keep the epoch that reaches the BEST
    value of the validation metric (maximum if accuracy, minimum if loss;
    "oracle early-stopping" style selection within the run itself).

    This collapses the "step"/epoch axis, leaving a single (val, test) point
    for each combination of run + hparams_seed + val_env.
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
    Select, SEPARATELY for each (dataset, algorithm, model, trial_seed,
    test_env), the winning hparams_seed.

    The score of each hparams_seed is the average (over the val_envs
    available for that test_env) of the best validation value reached at any
    epoch (see select_best_step_per_run). The winner is the one with the
    highest score if maximizing, or the lowest score if minimizing (loss).

    Only hparams_seeds that have results for the COMPLETE set of val_envs
    expected for that (trial_seed, test_env) are compared; if an hparams_seed
    has fewer val_envs than others, it is discarded to keep the comparison fair.
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

    # "Expected" set of val_envs for each (trial_seed, test_env): the union
    # of the val_envs seen across all candidate hparams_seeds.
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
    Retrieve, for each (dataset, algorithm, model, trial_seed, test_env),
    the test results corresponding to the winning hparams_seed, averaging
    over the val_envs that took part in that selection.

    Each trial_seed remains completely separate: it is never mixed with other
    trial_seeds at any point in the pipeline.
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
    """Convert a value into a safe folder name (no path separators)."""
    s = str(value)
    s = re.sub(r"[\\/]+", "_", s)
    s = s.strip().strip(".")
    return s or "unknown"


def save_results_by_group(test_summary, out_dir):
    """
    Save test_summary to out_dir/dataset/algorithm/model/best.csv, one file
    per (dataset, algorithm, model) combination. Folders are created
    automatically if they do not exist.
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
        help="jsonl file(s) or glob pattern(s)"
    )

    parser.add_argument(
        "--metric",
        default="loss",
        help=(
            "Metric to use as the selection and reporting criterion "
            "(default: loss). E.g.: acc, loss. Metrics whose name contains "
            "'loss' are minimized automatically."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "max", "min"],
        default="auto",
        help=(
            "Optimization direction. 'auto' (default) minimizes if the "
            "metric contains 'loss' and maximizes otherwise; 'max' and "
            "'min' force the direction."
        ),
    )

    parser.add_argument(
        "--test-envs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Restrict the output to these test domains. "
            "By default ALL observed test domains are computed."
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output folder (optional). Results are saved to "
            "<out>/<dataset>/<algorithm>/<model>/best.csv; folders are "
            "created automatically if they do not exist."
        ),
    )

    args = parser.parse_args()

    metric = args.metric
    minimize = is_minimized(metric, args.mode)
    direction = "minimum" if minimize else "maximum"

    val_col = f"val_{metric}"
    test_col = f"test_{metric}"
    mean_val_col = f"mean_{val_col}"
    std_val_col = f"std_{val_col}"
    mean_test_col = f"mean_{test_col}"
    std_test_col = f"std_{test_col}"

    print(f"[INFO] Metric: {metric} | selection criterion: {direction}")

    # Load records

    records = load_records(args.inputs)

    if not records:
        print("No records found.")
        return

    # Convert leave-two-out -> LODO protocol, for ALL possible
    # test domains

    df = build_lodo_dataframe(records, metric=metric)

    if df.empty:
        print(
            "No valid records found for the LODO protocol."
        )
        return

    if args.test_envs is not None:
        df = df[df["test_env"].isin(args.test_envs)].copy()
        if df.empty:
            print("No data for the requested --test-envs.")
            return

    # For each run, keep the epoch that gives the best value of the
    # validation metric (collapses the step/epoch axis)

    best_step_df = select_best_step_per_run(df, metric=metric, minimize=minimize)

    # Select hparams_seed using ONLY validation, separately
    # for each trial_seed and each test_env

    best_hparams, val_summary = compute_hparam_selection(
        best_step_df, metric=metric, minimize=minimize
    )

    print(
        f"\n=== Best hparams_seed per trial_seed and test domain "
        f"(selected using validation ONLY, {direction} per epoch) ===\n"
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

    # Evaluate that configuration on the test domain(s),
    # separately for each trial_seed

    test_summary = compute_test_results(
        best_step_df,
        best_hparams,
        metric=metric,
    )

    print(
        "\n=== Final result per trial_seed and test domain "
        "(each trial_seed is an independent experiment) ===\n"
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

    # Pivoted view: test domains as columns, for quick
    # comparison across trial_seeds

    pivot = test_summary.pivot_table(
        index=["dataset", "algorithm", "model", "trial_seed"],
        columns="test_env",
        values=mean_test_col,
    )
    pivot.columns = [f"test_env_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    print(
        "\n=== Pivoted view: mean_test_{metric} per test domain "
        "(columns) and trial_seed (rows) ===\n".format(metric=metric)
    )
    print(pivot.to_string(index=False))

    # Save: out/dataset/algorithm/model/best.csv

    if args.out:
        saved = save_results_by_group(test_summary_sorted, args.out)
        print(f"\nResults saved to {len(saved)} file(s):")
        for p in saved:
            print(f"  - {p}")


if __name__ == "__main__":
    main()