"""
Run a chosen anomaly detection method on the F2A datasets.

Data sources (in priority order):
  1. mTSBench/<DATASET>/<series>_train.csv + <series>_test.csv   (preferred)
  2. TSB-AD-M/<idx>_<DATASET>_id_*_tr_<N>_1st_*.csv               (fallback for
     datasets not in mTSBench: LTDB, SWaT, TAO)

Protocol (uniform across all 6 models, per project methodology):
  Fit on the train file (or data[:tr_N] for the TSB-AD-M fallback).
  Score and evaluate on the test file (or data[tr_N:]).
  Train and test never overlap.

Helpers receive arrays, not paths:
  anomaly_<MODEL>(data_train, data_test, labels_test, sliding_window)
                                                       -> (score, metrics)

Outputs (per model):
  results/<model>_per_file.csv      one row per (train,test) pair
  results/<model>_per_dataset.csv   one row per dataset, mean across pairs
  results/<model>_failures.csv      pairs that errored out

Usage:
    python run_model_all_datasets.py --model PCA \
        --mtsbench-dir ./mTSBench --tsbad-dir ./TSB-AD-M

    # Smoke test (1 series per dataset)
    python run_model_all_datasets.py --model PCA \
        --mtsbench-dir ./mTSBench --tsbad-dir ./TSB-AD-M --limit-per-dataset 1
"""

import argparse
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Auto-window detection (same convention as TSB-AD's main.py)
from TSB_AD.utils.slidingWindows import find_length_rank


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------
def _load_model(name: str):
    """Lazy-import the helper. Each helper exposes anomaly_<MODEL>(arrays...)."""
    if name == "PCA":
        from pca_helper import anomaly_PCA
        return anomaly_PCA
    if name == "IForest":
        from iforest_helper import anomaly_IForest
        return anomaly_IForest
    if name == "CBLOF":
        from cblof_helper import anomaly_CBLOF
        return anomaly_CBLOF
    if name == "RobustPCA":
        from robustpca_helper import anomaly_RobustPCA
        return anomaly_RobustPCA
    if name == "KMeansAD":
        from kmeansad_helper import anomaly_KMeansAD
        return anomaly_KMeansAD
    if name == "OmniAnomaly":
        from omnianomaly_helper import anomaly_OmniAnomaly
        return anomaly_OmniAnomaly
    raise ValueError(f"Unknown model: {name}")


AVAILABLE_MODELS = ["PCA", "IForest", "CBLOF", "RobustPCA", "KMeansAD", "OmniAnomaly"]


# -----------------------------------------------------------------------------
# F2A datasets and aliases
# -----------------------------------------------------------------------------
F2A_DATASETS = [
    "GECCO", "PSM", "Daphnet", "Genesis", "SWaT", "CreditCard",
    "GHL", "OPP", "SMAP", "MSL", "MITDB", "SVDB", "Exathlon",
    "SMD", "LTDB", "TAO",
]

# mTSBench folder name -> F2A canonical name. Case-insensitive.
MTSBENCH_FOLDER_ALIASES = {
    "OPPORTUNITY": "OPP",
    "OPP": "OPP",
    "creditcard": "CreditCard",
    "CreditCard": "CreditCard",
}

# TSB-AD-M filename token -> F2A canonical
TSBAD_TOKEN_ALIASES = {
    "OPPORTUNITY": "OPP",
}

# F2A datasets that mTSBench DOES NOT cover (-> use TSB-AD-M fallback)
TSBAD_FALLBACK_DATASETS = {"LTDB", "SWaT", "TAO"}

# TSB-AD-M filename pattern: <idx>_<DATASET>_id_..._tr_<N>_1st_<M>.csv
FNAME_RE = re.compile(r"^\d+_([A-Za-z0-9]+)_id_")


def _canonical_mtsbench(folder_name: str):
    """Map a folder name in mTSBench to an F2A canonical dataset name."""
    if folder_name in F2A_DATASETS:
        return folder_name
    return MTSBENCH_FOLDER_ALIASES.get(folder_name)


def _canonical_tsbad(token: str):
    """Map a TSB-AD-M filename token to an F2A canonical dataset name."""
    if token in F2A_DATASETS:
        return token
    return TSBAD_TOKEN_ALIASES.get(token)


# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------
def _parse_tr_index(filepath: Path) -> int:
    """Parse train-prefix length from TSB-AD-M filename: '..._tr_<N>_1st_<M>.csv'."""
    stem = filepath.stem
    return int(stem.split("_")[-3])


def discover_files(mtsbench_dir: Path, tsbad_dir: Path,
                   tsbad_eval_csv: Path | None = None):
    """Build {dataset: [task_dict, ...]} where each task represents one runnable
    (train, test) pair from either source.

    If tsbad_eval_csv is provided, only TSB-AD-M files in that list are used.
    """
    by_dataset = {d: [] for d in F2A_DATASETS}

    # Load the Eval-set whitelist once, if provided
    tsbad_eval_set = None
    if tsbad_eval_csv is not None:
        if not tsbad_eval_csv.exists():
            raise FileNotFoundError(
                f"TSB-AD-M Eval list not found at {tsbad_eval_csv}. "
                f"Pass --no-tsbad-eval-filter to skip filtering, or correct the "
                f"--tsbad-eval-csv path."
            )
        tsbad_eval_set = set(pd.read_csv(tsbad_eval_csv)["file_name"])


    # ---- mTSBench: pair _train.csv with _test.csv per subfolder ------------
    if mtsbench_dir.exists():
        for sub in sorted(mtsbench_dir.iterdir()):
            if not sub.is_dir():
                continue
            canon = _canonical_mtsbench(sub.name)
            if canon is None or canon not in by_dataset:
                continue
            # mTSBench is the preferred source, but for fallback datasets we
            # would have already chosen TSB-AD-M; here we just always take
            # whatever is available (fallback datasets are absent from mTSBench).
            for test_path in sorted(sub.glob("*_test.csv")):
                series = test_path.name[: -len("_test.csv")]
                train_path = sub / f"{series}_train.csv"
                if not train_path.exists():
                    # No matching train file -> this series isn't runnable
                    continue
                by_dataset[canon].append({
                    "source": "mtsbench",
                    "name": f"{canon}/{series}",
                    "mtsbench": {
                        "train_path": train_path,
                        "test_path": test_path,
                    },
                })

    # ---- TSB-AD-M fallback: only for datasets mTSBench doesn't cover -------
    if tsbad_dir.exists():
        for fp in sorted(tsbad_dir.glob("*.csv")):
            # skip files not in the Eval whitelist
            if tsbad_eval_set is not None and fp.name not in tsbad_eval_set:
                continue

            m = FNAME_RE.match(fp.name)
            if not m:
                continue
            canon = _canonical_tsbad(m.group(1))
            if canon is None:
                continue
            # Only use TSB-AD-M when (a) explicitly a fallback dataset, AND
            # (b) we don't already have anything for it from mTSBench.
            # The `and` is belt-and-braces; LTDB/SWaT/TAO aren't in mTSBench
            # so by_dataset[canon] will be empty for them anyway.
            if canon not in TSBAD_FALLBACK_DATASETS:
                continue
            try:
                tr_index = _parse_tr_index(fp)
            except (IndexError, ValueError):
                continue
            by_dataset[canon].append({
                "source": "tsbad",
                "name": f"{canon}/{fp.name}",
                "tsbad": {
                    "filepath": fp,
                    "tr_index": tr_index,
                },
            })

    return by_dataset


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
NON_FEATURE_COLUMNS = {"timestamp", "Timestamp", "time", "Time",
                        "Label", "label", "is_anomaly"}


def _load_csv(filepath: Path):
    """Load one CSV. Drops timestamp, separates features from labels.

    Returns (features: ndarray (n, d), labels: ndarray (n,) or None).
    Labels may be None if the file has no recognized label column (e.g.,
    some mTSBench train files contain only normal data without labels).
    """
    df = pd.read_csv(filepath).dropna()

    # Pick label column if present
    label_col = None
    for c in ("Label", "label", "is_anomaly"):
        if c in df.columns:
            label_col = c
            break

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    data = df[feature_cols].values.astype(float)
    labels = df[label_col].astype(int).to_numpy() if label_col is not None else None
    return data, labels


def _load_task(task: dict):
    """Resolve a task dict into (data_train, data_test, labels_test).

    For mtsbench: load both files directly.
    For tsbad:    load the single file, split at tr_index.
    """
    if task["source"] == "mtsbench":
        train_path = task["mtsbench"]["train_path"]
        test_path = task["mtsbench"]["test_path"]
        data_train, _ = _load_csv(train_path)
        data_test, labels_test = _load_csv(test_path)
        if labels_test is None:
            raise ValueError(f"Test file has no label column: {test_path.name}")
        return data_train, data_test, labels_test

    # tsbad source
    fp = task["tsbad"]["filepath"]
    tr_index = task["tsbad"]["tr_index"]
    data, labels = _load_csv(fp)
    if labels is None:
        raise ValueError(f"TSB-AD-M file has no label column: {fp.name}")
    if tr_index <= 0 or tr_index >= len(data):
        raise ValueError(
            f"tr_index={tr_index} out of bounds for length {len(data)} "
            f"in {fp.name}"
        )
    return data[:tr_index], data[tr_index:], labels[tr_index:]


def _auto_sliding_window(data: np.ndarray) -> int:
    """Auto-detect sliding window length for range-based metrics."""
    if data.shape[1] == 1:
        return find_length_rank(data, rank=1)
    return find_length_rank(data[:, 0].reshape(-1, 1), rank=1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS)
    parser.add_argument("--mtsbench-dir", type=Path, default=Path("./mTSBench"),
                        help="Root directory containing mTSBench dataset subfolders.")
    parser.add_argument("--tsbad-dir", type=Path, default=Path("./TSB-AD-M"),
                        help="Directory of TSB-AD-M flat CSVs (used for "
                             "datasets not in mTSBench: LTDB, SWaT, TAO).")
    parser.add_argument("--out-dir", type=Path, default=Path("./results"))
    parser.add_argument("--limit-per-dataset", type=int, default=None,
                        help="For debugging: only run first N series per dataset.")
    
    parser.add_argument("--tsbad-eval-csv", type=Path,
                    default=Path("./TSB-AD-M-Eva.csv"),
                    help="Path to TSB-AD-M-Eva.csv (the Eval-set file list). "
                         "TSB-AD-M files not in this list are skipped.")
    
    parser.add_argument("--no-tsbad-eval-filter", action="store_true",
                    help="Disable the Eval-set filter; use ALL TSB-AD-M files "
                         "matching the fallback datasets.")

    args = parser.parse_args()

    detector_fn = _load_model(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.model.lower()
    per_file_path = args.out_dir / f"{tag}_per_file.csv"
    per_dataset_path = args.out_dir / f"{tag}_per_dataset.csv"
    failures_path = args.out_dir / f"{tag}_failures.csv"
    if per_file_path.exists():
        per_file_path.unlink()

    tasks_by_dataset = discover_files(
    args.mtsbench_dir,
    args.tsbad_dir,
    tsbad_eval_csv=None if args.no_tsbad_eval_filter else args.tsbad_eval_csv,
)
    print(f"\n=== Model: {args.model} ===\n")
    print("Series discovered per dataset:")
    for d in F2A_DATASETS:
        n = len(tasks_by_dataset[d])
        if n == 0:
            marker = "!!"
        else:
            src = tasks_by_dataset[d][0]["source"]
            marker = f"[{src[:4]}]"
        print(f"  {marker:<6} {d:<12} {n:>4}")

    missing = [d for d in F2A_DATASETS if not tasks_by_dataset[d]]
    if missing:
        print(f"\nWARNING: no series for {missing}")
        print("(Check --mtsbench-dir / --tsbad-dir paths.)\n")

    total = sum(min(len(v), args.limit_per_dataset or 10**9)
                for v in tasks_by_dataset.values())
    print(f"\nTotal tasks: {total}\n")
    if total == 0:
        return

    per_file_rows = []
    failure_rows = []
    wrote_header = False

    pbar = tqdm(total=total, desc=args.model, unit="series")
    for dataset in F2A_DATASETS:
        tasks = tasks_by_dataset[dataset]
        if args.limit_per_dataset:
            tasks = tasks[:args.limit_per_dataset]

        for task in tasks:
            pbar.set_postfix_str(task["name"][:60])
            try:
                data_train, data_test, labels_test = _load_task(task)
                sliding_window = _auto_sliding_window(data_test)

                t0 = time.time()
                _, metrics = detector_fn(
                    data_train, data_test, labels_test, sliding_window
                )
                dt = time.time() - t0
                metrics = {k: float(v) for k, v in dict(metrics).items()}

                row = {
                    "model": args.model,
                    "dataset": dataset,
                    "source": task["source"],
                    "series": task["name"],
                    "runtime_sec": round(dt, 2),
                    "n_train": int(data_train.shape[0]),
                    "n_test": int(data_test.shape[0]),
                    "sliding_window": int(sliding_window),
                    **metrics,
                }
                per_file_rows.append(row)

                # Stream to CSV so a crash mid-run doesn't lose progress
                pd.DataFrame([row]).to_csv(
                    per_file_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True

            except Exception as e:
                failure_rows.append({
                    "model": args.model,
                    "dataset": dataset,
                    "series": task["name"],
                    "error": f"{type(e).__name__}: {e}",
                    "trace_tail": traceback.format_exc().splitlines()[-1],
                })
                tqdm.write(f"  [FAIL] {task['name']}: {type(e).__name__}: {e}")

            pbar.update(1)
    pbar.close()

    if per_file_rows:
        df_files = pd.DataFrame(per_file_rows)
        numeric_cols = df_files.select_dtypes(include=[np.number]).columns.tolist()
        # Drop sample-count columns from the per-dataset means
        for c in ("n_train", "n_test", "sliding_window"):
            if c in numeric_cols:
                numeric_cols.remove(c)

        df_summary = (
            df_files.groupby("dataset")[numeric_cols]
                    .mean()
                    .reindex(F2A_DATASETS)
                    .round(4)
        )
        df_summary.insert(
            0, "n_series",
            df_files.groupby("dataset").size().reindex(F2A_DATASETS).fillna(0).astype(int),
        )
        df_summary.to_csv(per_dataset_path)

        print(f"\n[OK] Per-file -> {per_file_path}  ({len(df_files)} rows)")
        print(f"[OK] Per-dataset -> {per_dataset_path}\n")
        print(f"=== {args.model}: per-dataset means ===")
        priority = ["VUS-PR", "AUC-PR", "AUC-ROC", "Standard-F1"]
        cols = ["n_series"] + [c for c in priority if c in df_summary.columns]
        print(df_summary[cols].to_string())
    else:
        print("\n[WARN] No tasks completed successfully.")

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
        print(f"\n[!] {len(failure_rows)} task(s) failed -> {failures_path}")


if __name__ == "__main__":
    main()