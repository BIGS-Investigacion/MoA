# LODO validation protocol

These scripts reproduce the leave-one-domain-out (LODO) model-selection protocol of DomainBed, while avoiding runs that would never be selected. Instead of training every combination of 1 and 2 held-out domains, the sweep only trains the combinations of **2** held-out domains (`--lodoval`). Each of those runs can then play two roles:

- held-out pair `[a, b]` → `val = a, test = b`
- held-out pair `[a, b]` → `val = b, test = a`

so a single run provides validation and test results for both choices of test domain.

Every `trial_seed` is treated as an independent experiment throughout the pipeline (hyperparameters are selected separately for each one). `seed` is only metadata and is never used for grouping.

## Pipeline overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `scripts/sweep.sh` | – | `sweep_output/**/results.jsonl` |
| 1 | `1-get_best_lodoval.py` | sweep logs | `best.csv` with the best `hparams_seed` per (dataset, algorithm, model, trial_seed, test_env) |
| 2 | `2-run_best_hparams.py` | `best.csv` | final runs (local or Slurm) |
| 3 | `3-get_final_results.py` | final logs | one row per run, at a fixed step (table + optional CSV) |
| 4 | `4-get_table.py` | results CSV | LaTeX table |

---

## 0. Run the sweep

```bash
bash scripts/sweep.sh "$Name" "$DATA_DIR" "$Dataset" "$Model" \
    "$N_hparams_from" "$N_hparams" "$Seed" "$N_Trials" 0
```

Positional arguments:

| # | Argument | Default | Meaning |
|---|----------|---------|---------|
| 1 | `name` | – | Experiment name (positional argument of `train_all.py`) |
| 2 | `path` | `domainbed/data` | Dataset root directory (`--data_dir`) |
| 3 | `dataset` | `PACS` | Dataset |
| 4 | `model` | `conch_moek_every_qkv` | Model |
| 5 | `n_hparams_from` | `0` | First `hparams_seed` (inclusive) |
| 6 | `n_hparams` | `20` | End of the `hparams_seed` range (**exclusive**): seeds `from … n_hparams-1` are run |
| 7 | `seed` | `42` | Base seed |
| 8 | `n_trials` | `3` | Number of trials (`trial_seed` = `0 … n_trials-1`) |
| 9 | `gpu_id` | `0` | GPU to use (`CUDA_VISIBLE_DEVICES`) |

Each job runs:

```bash
python train_all.py $name --data_dir $path --output_dir $output_dir \
    --algorithm ERM --dataset $dataset --model $model --l_aux \
    --seed $seed --trial_seed $trial --hparams_seed $hparams_seed \
    --es_val=-1 --lodoval --holdout_fraction=0.0
```

Notes:

- `--lodoval` makes every run hold out a pair of domains `[a, b]`, recorded in `args.real_test_envs`. `--holdout_fraction=0.0` means no extra `out` split is created.
- Jobs run **sequentially** on a single GPU. To use several GPUs, launch several instances of the script with different `n_hparams_from` / `n_hparams` ranges and different `gpu_id`.
- `seed` is incremented after every job, so it differs between runs. It is metadata only.
- A finished job writes a `done` file in its output directory. Re-running the script skips jobs that already have it, so the sweep can be resumed. Failed jobs are reported as `FAILED` and are relaunched on the next execution.

Expected layout of the output: `sweep_output/NEW/<dataset>/<model>/<name>/MoA_sweep_h_<hparams_seed>_t_<trial>/results.jsonl`.

## 1. Get the best hyperparameters

For every (dataset, algorithm, model, trial_seed, test_env), this step picks the `hparams_seed` with the best validation score, using **validation data only**:

1. Within each run, keep the epoch with the best validation metric.
2. Average that value over the available validation domains.
3. Choose the `hparams_seed` with the best average.

Only `hparams_seed` values that have results for the complete set of validation domains are compared, to keep the comparison fair.

```bash
python 1-get_best_lodoval.py \
    "sweep_output/**/results.jsonl" \
    --metric loss \
    --out Best_Hparams
```

Notes:

- **Quote the glob** so `**` is expanded by Python (recursive) and not by your shell.
- Metrics whose name contains `loss` are minimized; the rest are maximized. Use `--mode min|max` to force the direction. The default metric is `acc`.
- `--test-envs` restricts the output to some test domains.
- Results are written to `Best_Hparams/<dataset>/<algorithm>/<model>/best.csv`, one file per (dataset, algorithm, model).
- The script also prints the selected `hparams_seed`, the final results per `trial_seed` and test domain, and a pivoted view (test domains as columns).
- Step 2 only needs the columns `dataset`, `algorithm`, `model`, `trial_seed`, `test_env` and `hparams_seed`.

## 2. Final runs with the best hyperparameters

Train again with the selected hyperparameters. By default `--lodoval` is **not** passed, so each run is trained on **all domains except `test_env`** (`--test_envs <test_env>`), which is the setting you want for the final results. Add `--lodoval` only if you want to re-run the leave-two-out mode of the sweep (not needed in the normal pipeline).

One `train_all.py` run is launched for each unique (dataset, algorithm, model, trial_seed, hparams_seed, test_env). If two rows of the CSV request exactly the same run, it is executed only once.

`run_best_hparams_moa.py` takes a single CSV. Since step 1 writes one `best.csv` per (dataset, algorithm, model), run this step once per file (or concatenate them first).

Use `--output-dir FINAL_RESULTS` so that step 3 finds the logs with its default `--data_dir`.

### On Slurm (one `sbatch` per run)

Run it from the login node:

```bash
python 2-run_best_hparams.py "$csv_path" \
    --name "$Name" \
    --data-dir "$DATA_DIR" \
    --output-dir FINAL_RESULTS \
    --slurm \
    --partition gpu \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem 32G \
    --time 12:00:00
```

Slurm-specific options: `--job-prefix`, `--log-dir` (default `slurm_logs`), `--account`, `--sbatch-extra "<extra sbatch args>"`, and `--setup-cmd` (commands run before training inside each job; by default it activates the `MoA` conda environment).

### Locally (no Slurm)

Runs are executed sequentially with the current Python interpreter (`--python` to change it).

```bash
python run_best_hparams_moa.py "$csv_path" \
    --name "$Name" \
    --data-dir "$DATA_DIR" \
    --output-dir FINAL_RESULTS
```

Notes:

- Add `--dry-run` to print the commands (or the `sbatch` calls) without running anything.
- `--stop-on-error` stops at the first failure (default: keep going and report the failures at the end).
- Filters (`--datasets`, `--algorithms`, `--models`, `--trial-seeds`, `--test-envs`) restrict which rows of the CSV are launched.
- Other options passed through to `train_all.py`: `--seed` (default `1`; it is not in the CSV), `--es-val` (default `6`, whereas the sweep used `-1`), `--no-l-aux` (to drop `--l_aux`), `--model` (to force the model name) and `--extra "<args>"` for anything else.
- There is no `--skip-existing`: re-running the script launches every selected run again.

## 3. Get final results

Reads every `results.jsonl` under `--data_dir`, keeps the records logged at `--step`, and reports for each run the test metrics of its test domain (`env{test_env}_in_{metric}`), together with the `hparams_seed` used. Rows are indexed by (dataset, trial_seed, test_env, model, algorithm).

```bash
python 3-get_final_results.py \
    --data_dir FINAL_RESULTS \
    --step 1500 \
    --out Final_Results/results.csv
```

Notes:

- `--step` must be a step that the final runs actually log (default `1500`).
- `--metrics` selects which metrics to report (default: `acc bacc f1 f1_pos0 f1_pos1 auroc`).
- The script expects logs with a **single** test environment, i.e. the final runs of step 2 without `--lodoval`. It fails if it finds logs with more than one, so do not point it at the sweep output.
- If several rows share the same index (e.g. a run was executed twice), a warning is printed with the repeated rows.
- `--out` is optional; the parent folder is created if it does not exist.

## 4. Get summary table

Averages over test domains within each seed, then aggregates over seeds (mean or median ± standard error) and writes a LaTeX table with the best result per dataset in bold. Requires `\usepackage{booktabs}` and `\usepackage{multirow}`.

```bash
python 4-get_table.py \
    --csv Final_Results/results.csv \
    --out Final_Results/table.tex
```

Useful options: `--agg mean|median`, `--scale 100` (percentages), `--decimals 1`, `--caption`, `--label`.