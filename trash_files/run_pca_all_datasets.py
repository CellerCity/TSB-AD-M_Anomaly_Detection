"""
Run PCA anomaly detection on all 16 F2A datasets in TSB-AD-M.

Outputs:
  - results/pca_per_file.csv      : one row per file, all metrics
  - results/pca_per_dataset.csv   : one row per dataset, mean of metrics across files
  - results/pca_failures.csv      : files that errored out, with reason

Usage:
    python run_pca_all_datasets.py --data-dir ./TSB-AD-M --out-dir ./results
"""

import argparse
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd

from pca_helper import anomaly_PCA


# 16 datasets from F2A paper (Table 1 / Table 2).
# The dataset *token* is what appears in the TSB-AD-M filename between the index
# and `_id_`. We match on this token so e.g. "SMD" doesn't accidentally match
# files for "MITDB" or "SVDB".
# F2A_DATASETS = [
#     "GECCO",
#     "PSM",
#     "Daphnet",
#     "Genesis",
#     "SWaT",
#     "CreditCard",
#     "GHL",
#     "OPPORTUNITY",          
#     "SMAP",
#     "MSL",
#     "MITDB",
#     "SVDB",
#     "Exathlon",
#     "SMD",
#     "LTDB",
#     "TAO",
# ]

F2A_DATASETS = [
    # "GECCO",
    # "PSM",
    # "Daphnet",
    # "TAO",
    'Exathlon'
]



# Filename format from TSB-AD-M, e.g.
#   174_Exathlon_id_1_Facility_tr_10766_1st_12590.csv
#   001_GECCO_id_1_Sensor_tr_69261_1st_71561.csv
# Pattern: <index>_<DATASET>_id_...
FNAME_RE = re.compile(r"^\d+_([A-Za-z0-9]+)_id_")

eval_files = set(pd.read_csv("./File_List/TSB-AD-M-Eva.csv")["file_name"])

def discover_files(data_dir: Path, eval_only=True):
    """Return {dataset_token: [list of Paths]} for the 16 F2A datasets."""
    by_dataset = {d: [] for d in F2A_DATASETS}

    all_csvs = sorted(data_dir.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    for fp in all_csvs:
        if eval_only and fp.name not in eval_files:
            continue   # skip Tuning files
        m = FNAME_RE.match(fp.name)
        if not m:
            continue
        token = m.group(1)
        if token in by_dataset:
            by_dataset[token].append(fp)

    return by_dataset


def run_one_file(fp: Path):
    """Run PCA on a single file. Returns (metrics_dict, runtime_seconds) or raises."""
    t0 = time.time()
    anomaly_score, metrics = anomaly_PCA(str(fp))
    dt = time.time() - t0

    # `metrics` is the dict returned by TSB-AD's get_metrics. Coerce to plain
    # dict of float so it serializes cleanly.
    metrics = {k: float(v) for k, v in dict(metrics).items()}
    return metrics, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing TSB-AD-M CSV files")
    parser.add_argument("--out-dir", default=Path("./results"), type=Path,
                        help="Where to write result CSVs")
    parser.add_argument("--limit-per-dataset", type=int, default=None,
                        help="For debugging: only run first N files per dataset")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_file_path = args.out_dir / "pca_per_file.csv"
    per_dataset_path = args.out_dir / "pca_per_dataset.csv"
    failures_path = args.out_dir / "pca_failures.csv"

    files_by_dataset = discover_files(args.data_dir)

    # Report what we found before running anything
    print("\n=== Files discovered per dataset ===")
    for d in F2A_DATASETS:
        n = len(files_by_dataset[d])
        marker = "  " if n > 0 else "!!"
        print(f"  {marker} {d:<12} {n:>4} file(s)")
    missing = [d for d in F2A_DATASETS if not files_by_dataset[d]]
    if missing:
        print(f"\nWARNING: no files found for: {missing}")
        print("Check your --data-dir or naming convention.\n")

    total_files = sum(min(len(v), args.limit_per_dataset or 10**9)
                      for v in files_by_dataset.values())
    print(f"\nTotal files to process: {total_files}\n")

    # Stream results to disk as we go so a crash doesn't lose everything
    per_file_rows = []
    failure_rows = []
    wrote_header = False

    pbar = tqdm(total=total_files, desc="PCA over files", unit="file")
    for dataset in F2A_DATASETS:
        files = files_by_dataset[dataset]
        if args.limit_per_dataset:
            files = files[:args.limit_per_dataset]

        for fp in files:
            pbar.set_postfix_str(f"{dataset}/{fp.name[:40]}")
            try:
                metrics, runtime = run_one_file(fp)
                row = {
                    "dataset": dataset,
                    "filename": fp.name,
                    "runtime_sec": round(runtime, 2),
                    **metrics,
                }
                per_file_rows.append(row)

                # Append to CSV incrementally
                df_row = pd.DataFrame([row])
                df_row.to_csv(
                    per_file_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True

            except Exception as e:
                failure_rows.append({
                    "dataset": dataset,
                    "filename": fp.name,
                    "error": f"{type(e).__name__}: {e}",
                    "trace_tail": traceback.format_exc().splitlines()[-1],
                })
                tqdm.write(f"  [FAIL] {fp.name}: {type(e).__name__}: {e}")

            pbar.update(1)

    pbar.close()

    # Final tables
    if per_file_rows:
        df_files = pd.DataFrame(per_file_rows)
        df_files.to_csv(per_file_path, index=False)
        print(f"\n[OK] Per-file results -> {per_file_path}  ({len(df_files)} rows)")

        # Per-dataset means. Numeric columns only.
        numeric_cols = df_files.select_dtypes(include=[np.number]).columns.tolist()
        df_summary = (
            df_files
            .groupby("dataset")[numeric_cols]
            .mean()
            .reindex(F2A_DATASETS)        # keep paper's row order
            .round(4)
        )
        # Add file count column
        df_summary["n_files"] = df_files.groupby("dataset").size().reindex(F2A_DATASETS).fillna(0).astype(int)
        df_summary.to_csv(per_dataset_path)
        print(f"[OK] Per-dataset means -> {per_dataset_path}\n")

        # Pretty print the summary, focused on the two metrics F2A reports
        print("=== Per-dataset means (key metrics) ===")
        cols_to_show = [c for c in ["VUS-PR", "VUS_PR", "VUS-ROC", "AUC-PR", "AUC-ROC", "Standard-F1", "n_files"]
                        if c in df_summary.columns]
        print(df_summary[cols_to_show].to_string())
    else:
        print("\n[WARN] No files processed successfully.")

    if failure_rows:
        df_fail = pd.DataFrame(failure_rows)
        df_fail.to_csv(failures_path, index=False)
        print(f"\n[!] {len(df_fail)} file(s) failed -> {failures_path}")


if __name__ == "__main__":
    main()