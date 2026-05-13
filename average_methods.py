import argparse
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-file metric CSVs from one or more ablation runs into a single summary."
    )

    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="List of per-run CSVs to summarize (one row per file inside each)."
    )
    parser.add_argument(
        "--output", type=str, default="final_method_averages.csv",
        help="Output summary CSV (one row per input)."
    )

    # Optional parallel lists of context/horizon labels — one per input file.
    # If supplied, must match len(inputs).
    parser.add_argument(
        "--contexts", nargs="*", type=int, default=None,
        help="(Optional) context_length per input file, in the same order as --inputs."
    )
    parser.add_argument(
        "--horizons", nargs="*", type=int, default=None,
        help="(Optional) horizon per input file, in the same order as --inputs."
    )
    parser.add_argument(
        "--score_methods", nargs="*", type=str, default=None,
        help="(Optional) score_method per input file, in the same order as --inputs."
    )

    args = parser.parse_args()

    # Validate parallel-list lengths if provided.
    for name, vals in [("contexts", args.contexts),
                       ("horizons", args.horizons),
                       ("score_methods", args.score_methods)]:
        if vals is not None and len(vals) != len(args.inputs):
            raise ValueError(
                f"--{name} has {len(vals)} entries but --inputs has {len(args.inputs)}; "
                f"they must match."
            )

    summary_data = []
    print(f"Processing {len(args.inputs)} files...")

    for idx, filepath in enumerate(args.inputs):
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            if len(df) == 0:
                print(f"[WARNING] {filepath} is empty, skipping")
                continue

            label = os.path.basename(filepath).replace(".csv", "")
            row = {
                "Source_File":           os.path.basename(filepath),
                "Label":                 label,
                "Total_Files_Evaluated": len(df),
                "Mean_AUROC":            df["AUROC"].mean(),
                "Mean_AUPRC":            df["AUPRC"].mean(),
                "Mean_VUS-ROC":          df["VUS-ROC"].mean(),
                "Mean_VUS-PR":           df["VUS-PR"].mean(),
            }
            if args.contexts      is not None: row["Context"]      = args.contexts[idx]
            if args.horizons      is not None: row["Horizon"]      = args.horizons[idx]
            if args.score_methods is not None: row["Score_Method"] = args.score_methods[idx]

            print(f"\n{label}  (files={row['Total_Files_Evaluated']})")
            print(f"  AUROC:   {row['Mean_AUROC']:.4f}")
            print(f"  AUPRC:   {row['Mean_AUPRC']:.4f}")
            print(f"  VUS-ROC: {row['Mean_VUS-ROC']:.4f}")
            print(f"  VUS-PR:  {row['Mean_VUS-PR']:.4f}")

            summary_data.append(row)

        except Exception as e:
            print(f"[ERROR] Could not process {filepath}: {e}")

    if not summary_data:
        print("\n[ERROR] No valid data was processed.")
        return

    new_summary_df = pd.DataFrame(summary_data)

    # Order columns: identifying labels first, then metrics.
    label_cols  = [c for c in ["Context", "Horizon", "Score_Method",
                               "Label", "Source_File", "Total_Files_Evaluated"]
                   if c in new_summary_df.columns]
    metric_cols = ["Mean_AUROC", "Mean_AUPRC", "Mean_VUS-ROC", "Mean_VUS-PR"]
    new_summary_df = new_summary_df[label_cols + metric_cols]

    if os.path.exists(args.output):
        existing_df  = pd.read_csv(args.output)
        combined_df  = pd.concat([existing_df, new_summary_df], ignore_index=True)
        # Dedupe on source filename, keeping the latest row.
        combined_df  = combined_df.drop_duplicates(subset=["Source_File"], keep="last")
    else:
        combined_df = new_summary_df

    combined_df.to_csv(args.output, index=False)
    print("\n" + "=" * 50)
    print(f"[SUCCESS] Summary saved to: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()


# python average_methods.py --inputs smd_c512_h128.csv --contexts 512 --horizons 128 --output sweep.csv