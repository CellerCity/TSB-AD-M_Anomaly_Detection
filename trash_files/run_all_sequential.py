"""
Sequential runner for TSB-AD-M anomaly detection benchmark.

One process, one model at a time, one file at a time. No parallelism,
no subprocesses, no GPU pool. Just loops.

Outputs per model:
    results/<model>_per_file.csv      one row per file
    results/<model>_per_dataset.csv   one row per dataset (mean across files)
    results/<model>_failures.csv      files that errored out

Usage:
    # Default: all 6 models, Eval-only filter
    python run_all_sequential.py --data-dir ./TSB-AD-M

    # Subset of models
    python run_all_sequential.py --data-dir ./TSB-AD-M --models PCA IForest

    # Smoke test: 1 file per dataset
    python run_all_sequential.py --data-dir ./TSB-AD-M --limit-per-dataset 1

    # Run on all 200 files, not just Eval
    python run_all_sequential.py --data-dir ./TSB-AD-M --no-eval-filter
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


# ---- Model registry ---------------------------------------------------------
def _load_model(name: str):
    """Lazy-import the helper. Same as launcher.py / run_one_file.py."""
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


ALL_MODELS = ["PCA", "IForest", "CBLOF", "RobustPCA", "KMeansAD", "OmniAnomaly"]


# ---- Dataset discovery ------------------------------------------------------

F2A_DATASETS = [
    "GECCO", "PSM", "Daphnet", "Genesis", "SWaT", "CreditCard",
    "GHL", "OPP", "SMAP", "MSL", "MITDB", "SVDB", "Exathlon",
    "SMD", "LTDB", "TAO",
]

# TSB-AD-M filenames may use "OPPORTUNITY" while F2A's table uses "OPP".
DATASET_ALIASES = {
    "OPP": ["OPP", "OPPORTUNITY", "Opportunity"],
}

FNAME_RE = re.compile(r"^\d+_([A-Za-z0-9]+)_id_")


def _canonical_dataset(token: str):
    if token in F2A_DATASETS:
        return token
    for canon, aliases in DATASET_ALIASES.items():
        if token in aliases:
            return canon
    return None


def discover_files(data_dir: Path, eval_only: bool, eval_csv_path: Path):
    """Return {dataset: [Paths]} restricted to F2A datasets and (optionally) Eval set."""
    by_dataset = {d: [] for d in F2A_DATASETS}

    eval_files = None
    if eval_only:
        if not eval_csv_path.exists():
            raise FileNotFoundError(
                f"Eval list not found at {eval_csv_path}. "
                f"Pass --eval-csv or use --no-eval-filter."
            )
        eval_files = set(pd.read_csv(eval_csv_path)["file_name"])

    all_csvs = sorted(data_dir.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for fp in all_csvs:
        if eval_only and fp.name not in eval_files:
            continue
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        canon = _canonical_dataset(m.group(1))
        if canon is not None:
            by_dataset[canon].append(fp)

    return by_dataset


# ---- Per-model run ----------------------------------------------------------

def run_one_model(model: str, files_by_dataset: dict, args, out_dir: Path):
    """Run a single model over all discovered files, write outputs, return summary df."""
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")

    # Resolve helper
    try:
        detector_fn = _load_model(model)
    except Exception as e:
        print(f"  [SKIP] Cannot import helper for {model}: {e}")
        return None

    tag = model.lower()
    per_file_path = out_dir / f"{tag}_per_file.csv"
    per_dataset_path = out_dir / f"{tag}_per_dataset.csv"
    failures_path = out_dir / f"{tag}_failures.csv"

    # Wipe stale incremental files
    for p in [per_file_path, failures_path]:
        if p.exists():
            p.unlink()

    # Count tasks
    total = sum(min(len(v), args.limit_per_dataset or 10**9)
                for v in files_by_dataset.values())
    if total == 0:
        print("  No files to process.")
        return None

    per_file_rows = []
    failure_rows = []
    wrote_header = False

    pbar = tqdm(total=total, desc=model, unit="file", leave=True)
    for dataset in F2A_DATASETS:
        files = files_by_dataset[dataset]
        if args.limit_per_dataset:
            files = files[:args.limit_per_dataset]

        for fp in files:
            pbar.set_postfix_str(f"{dataset}/{fp.name[:35]}")
            try:
                t0 = time.time()
                _, metrics = detector_fn(str(fp))
                dt = time.time() - t0
                metrics = {k: float(v) for k, v in dict(metrics).items()}

                row = {
                    "model": model,
                    "dataset": dataset,
                    "filename": fp.name,
                    "runtime_sec": round(dt, 2),
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
                    "model": model,
                    "dataset": dataset,
                    "filename": fp.name,
                    "error": f"{type(e).__name__}: {e}",
                    "trace_tail": traceback.format_exc().splitlines()[-1],
                })
                tqdm.write(f"  [FAIL] {fp.name}: {type(e).__name__}: {e}")

            pbar.update(1)
    pbar.close()

    # Per-dataset summary
    summary = None
    if per_file_rows:
        df = pd.DataFrame(per_file_rows)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary = (
            df.groupby("dataset")[numeric_cols]
              .mean()
              .reindex(F2A_DATASETS)
              .round(4)
        )
        summary.insert(
            0, "n_files",
            df.groupby("dataset").size().reindex(F2A_DATASETS).fillna(0).astype(int),
        )
        summary.to_csv(per_dataset_path)

        # Pretty print
        print(f"\n  --- {model} per-dataset means ---")
        priority = ["VUS-PR", "VUS_PR", "AUC-PR", "AUC-ROC", "Standard-F1"]
        cols = ["n_files"] + [c for c in priority if c in summary.columns]
        print(summary[cols].to_string())

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(failures_path, index=False)
        print(f"\n  [!] {len(failure_rows)} file(s) failed -> {failures_path}")

    return summary


# ---- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("./results"), type=Path)
    parser.add_argument("--eval-csv", default=Path("./TSB-AD-M-Eva.csv"), type=Path)
    parser.add_argument("--no-eval-filter", action="store_true",
                        help="Run on all files in --data-dir, ignoring the Eval list.")
    parser.add_argument("--limit-per-dataset", type=int, default=None,
                        help="For debugging: only run first N files per dataset.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help=f"Models to run (default: all {len(ALL_MODELS)}).")
    args = parser.parse_args()

    # Validate model names
    for m in args.models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model: {m}. Available: {ALL_MODELS}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Discover files once, reuse for all models
    files_by_dataset = discover_files(
        args.data_dir,
        eval_only=not args.no_eval_filter,
        eval_csv_path=args.eval_csv,
    )

    eval_label = "Eval-only (180 files)" if not args.no_eval_filter else "ALL files"
    print(f"\nSequential runner")
    print(f"  Models:   {args.models}")
    print(f"  Mode:     {eval_label}")
    print(f"\n  Files per dataset:")
    for d in F2A_DATASETS:
        n = len(files_by_dataset[d])
        marker = "  " if n > 0 else "!!"
        print(f"    {marker} {d:<12} {n:>4}")

    missing = [d for d in F2A_DATASETS if not files_by_dataset[d]]
    if missing:
        print(f"\n  WARNING: no files found for: {missing}")

    overall_t0 = time.time()
    for model in args.models:
        run_one_model(model, files_by_dataset, args, args.out_dir)

    print(f"\n{'='*60}")
    print(f"  Done. Total time: {(time.time() - overall_t0)/60:.1f} min")
    print(f"  Results in: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
