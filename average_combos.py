"""
Aggregate one or more "all combos" CSVs (output of timesFM_all_combos.py) into a
single summary table where each row is one (context, horizon, score_method, agg_method)
combination averaged across files.

Each input CSV is expected to have columns:
    file_name, context_length, horizon, score_method, agg_method,
    AUROC, AUPRC, VUS-ROC, VUS-PR
"""

import argparse
import os
import sys
import pandas as pd


REQUIRED_COLS = [
    "file_name", "context_length", "horizon",
    "score_method", "agg_method",
    "AUROC", "AUPRC", "VUS-ROC", "VUS-PR",
]


def main():
    parser = argparse.ArgumentParser(
        description="Average all_combos CSVs by (context, horizon, score, agg). "
                    "Each input row is (file, combo); each output row is one combo averaged over files."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="List of all_combos CSVs (or a single glob like 'ablation_results/*all.csv').",
    )
    parser.add_argument(
        "--output", type=str, default="ablation_combos_summary.csv",
        help="Output summary CSV path.",
    )
    parser.add_argument(
        "--sort_by", type=str, default="VUS-ROC",
        choices=["AUROC", "AUPRC", "VUS-ROC", "VUS-PR"],
        help="Metric to sort the output table by (descending).",
    )
    args = parser.parse_args()

    # -------------------- READ AND VALIDATE --------------------
    frames = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}")
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[ERROR] Could not read {path}: {e}")
            continue

        if len(df) == 0:
            print(f"[WARNING] {path} is empty, skipping")
            continue

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[ERROR] {path} missing columns: {missing}; skipping")
            continue

        df["_source"] = os.path.basename(path)
        frames.append(df)
        print(f"  loaded {path}  ({len(df)} rows)")

    if not frames:
        print("\n[ERROR] No valid input files. Nothing to do.")
        sys.exit(1)

    full = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows across all inputs: {len(full)}")

    # -------------------- AGGREGATE --------------------
    group_keys = ["context_length", "horizon", "score_method", "agg_method"]
    metric_cols = ["AUROC", "AUPRC", "VUS-ROC", "VUS-PR"]

    summary = (
        full.groupby(group_keys, as_index=False)
            .agg(
                n_files=("file_name", "nunique"),
                **{m: (m, "mean") for m in metric_cols},
            )
    )

    # -------------------- SORT AND SAVE --------------------
    summary = summary.sort_values(args.sort_by, ascending=False).reset_index(drop=True)
    summary.to_csv(args.output, index=False)

    # -------------------- PRETTY-PRINT --------------------
    print("\n" + "=" * 80)
    print(f"SUMMARY  ({len(summary)} unique combinations)")
    print("=" * 80)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 80)

    # Highlight the single best combination per metric.
    print("\nBest combination per metric:")
    for m in metric_cols:
        best = summary.sort_values(m, ascending=False).iloc[0]
        print(
            f"  {m:8s}: context={int(best['context_length'])} "
            f"horizon={int(best['horizon'])} "
            f"score={best['score_method']:22s} "
            f"agg={best['agg_method']:10s} "
            f"value={best[m]:.4f}  (avg over {int(best['n_files'])} files)"
        )

    print(f"\n[SUCCESS] Saved {len(summary)} rows to: {args.output}")


if __name__ == "__main__":
    main()


# Example usage:
#   python average_combos.py --inputs ablation_results/smd_c512_h128_all.csv
#   python average_combos.py --inputs ablation_results/*all.csv --output smd_all_summary.csv
#   python average_combos.py --inputs ablation_results/*all.csv --sort_by AUPRC