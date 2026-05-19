"""
Compile per-dataset results from all model runs into comparison tables.

Layout: rows = models, columns = datasets, one CSV per metric.

Reads:
    results/<model>_per_dataset.csv  for each model

Writes (in --out-dir):
    comparison_<METRIC>.csv  one table per metric (models x datasets)

Usage:
    # Default: all 6 models, all 16 datasets, all metrics
    python compile_results.py --results-dir ./results

    # Subset of datasets, models, or metrics
    python compile_results.py --datasets Exathlon TAO SMD
    python compile_results.py --models PCA IForest CBLOF
    python compile_results.py --metrics VUS-PR AUC-PR Standard-F1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# F2A Table 2 row order (now the COLUMN order, since we transposed)
F2A_DATASETS = [
    "GECCO", "PSM", "Daphnet", "Genesis", "SWaT", "CreditCard",
    "GHL", "OPP", "SMAP", "MSL", "MITDB", "SVDB", "Exathlon",
    "SMD", "LTDB", "TAO",
]

# F2A Table 2 column order (now the ROW order)
DEFAULT_MODELS = ["IForest", "CBLOF", "PCA", "RobustPCA", "KMeansAD", "OmniAnomaly", "FITS", "CNN"]

# Default metric order: F2A's primary first, then the rest
DEFAULT_METRIC_PRIORITY = [
    "VUS-PR", "AUC-PR", "VUS-ROC", "AUC-ROC",
    "Standard-F1", "PA-F1", "Event-based-F1", "R-based-F1",
    "Affiliation-F",
]


def _load_one_model(results_dir: Path, model: str):
    """Load <results>/<model_lower>_per_dataset.csv. Returns None if missing."""
    fp = results_dir / f"{model.lower()}_per_dataset.csv"
    if not fp.exists():
        return None
    return pd.read_csv(fp).set_index("dataset")


def build_comparison_table(per_model_dfs: dict, metric: str,
                            models: list, datasets: list) -> pd.DataFrame:
    """Build a models x datasets comparison table for one metric."""
    table = pd.DataFrame(
        index=models,
        columns=datasets,
        dtype=float,
    )
    table.index.name = "Model"

    for model in models:
        df = per_model_dfs.get(model)
        if df is None or metric not in df.columns:
            continue
        for ds in datasets:
            if ds in df.index:
                table.loc[model, ds] = df.loc[ds, metric]

    # Add a "Mean" column (mean across the selected datasets for each model)
    # table["Mean"] = table.mean(axis=1, skipna=True)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"),
                        help="Directory containing <model>_per_dataset.csv files.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write compiled tables. Defaults to --results-dir.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to include as rows, in order. "
                             "Default matches F2A Table 2 ordering.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to include as columns. "
                             "Default = all 16 F2A datasets.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metrics to include (one CSV per metric). "
                             "Default = all metrics found in the per-dataset CSVs.")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve datasets
    if args.datasets:
        unknown_ds = [d for d in args.datasets if d not in F2A_DATASETS]
        if unknown_ds:
            print(f"  [WARN] Datasets not in F2A list: {unknown_ds}")
        datasets = [d for d in args.datasets if d in F2A_DATASETS]
        if not datasets:
            print("  No valid datasets selected. Exiting.")
            return
    else:
        datasets = F2A_DATASETS

    # Load each model's per-dataset CSV
    per_model_dfs = {}
    missing = []
    for m in args.models:
        df = _load_one_model(args.results_dir, m)
        per_model_dfs[m] = df
        if df is None:
            missing.append(m)

    if missing:
        print(f"  [WARN] No per-dataset CSV for: {missing}")
        print(f"         Those rows will be empty.\n")

    # Discover available metrics across loaded CSVs
    available = set()
    for df in per_model_dfs.values():
        if df is not None:
            available.update(c for c in df.columns
                             if c not in {"n_files", "runtime_sec"})

    if args.metrics:
        metrics = [m for m in args.metrics if m in available]
        unknown = set(args.metrics) - available
        if unknown:
            print(f"  [WARN] Requested metrics not found in any CSV: {sorted(unknown)}")
    else:
        metrics = [m for m in DEFAULT_METRIC_PRIORITY if m in available]
        for m in sorted(available):
            if m not in metrics:
                metrics.append(m)

    if not metrics:
        print("  No metrics found. Are the per-dataset CSVs in the right place?")
        return

    print(f"\n  Models:    {args.models}")
    print(f"  Datasets:  {datasets}")
    print(f"  Metrics:   {metrics}")
    print(f"  Out dir:   {out_dir}\n")

    for metric in metrics:
        table = build_comparison_table(per_model_dfs, metric, args.models, datasets)
        csv_path = out_dir / f"comparison_{metric}.csv"
        table.to_csv(csv_path, float_format="%.4f")

        print(f"--- {metric} -> {csv_path} ---")
        print(table.round(4).to_string())
        print()

    print(f"  [OK] Wrote {len(metrics)} CSV(s) to {out_dir}")


if __name__ == "__main__":
    main()