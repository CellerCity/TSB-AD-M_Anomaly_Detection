import argparse
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate individual averages from multiple metric CSVs and combine them.")
    
    # Accept multiple files as input using nargs='+'
    parser.add_argument("--inputs", nargs='+', required=True, 
                        help="List of CSV files to average (e.g., --inputs smd_results_mse.csv smd_results_interval.csv)")
    
    parser.add_argument("--output", type=str, default="final_method_averages.csv", 
                        help="The final CSV file where the summary averages will be saved.")
    
    args = parser.parse_args()

    summary_data = []

    print(f"Processing {len(args.inputs)} files...")

    # Process each file individually
    for filepath in args.inputs:
        if not os.path.exists(filepath):
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        try:
            df = pd.read_csv(filepath)
            
            # Figure out the method name (use the column if it exists, otherwise use the filename)
            if 'Score_Method' in df.columns:
                method_name = df['Score_Method'].iloc[0]
            else:
                # e.g., "smd_results_mse.csv" -> "smd_results_mse"
                method_name = os.path.basename(filepath).replace('.csv', '')

            # Calculate the individual averages for this specific file
            num_files = len(df)
            avg_auroc = df['AUROC'].mean()
            avg_auprc = df['AUPRC'].mean()
            avg_vus_roc = df['VUS-ROC'].mean()
            avg_vus_pr = df['VUS-PR'].mean()

            print(f"\nMethod: {method_name.upper()} (From {os.path.basename(filepath)})")
            print(f"  Files Evaluated: {num_files}")
            print(f"  AUROC:   {avg_auroc:.4f}")
            print(f"  AUPRC:   {avg_auprc:.4f}")
            print(f"  VUS-ROC: {avg_vus_roc:.4f}")
            print(f"  VUS-PR:  {avg_vus_pr:.4f}")

            # Store the summarized row
            summary_data.append({
                "Method": method_name.upper(),
                "Source_File": os.path.basename(filepath),
                "Total_Files_Evaluated": num_files,
                "Mean_AUROC": avg_auroc,
                "Mean_AUPRC": avg_auprc,
                "Mean_VUS-ROC": avg_vus_roc,
                "Mean_VUS-PR": avg_vus_pr
            })

        except Exception as e:
            print(f"[ERROR] Could not process {filepath}: {e}")

    # Combine all summaries and export to the final file
    if summary_data:
        new_summary_df = pd.DataFrame(summary_data)

        # If the summary file already exists, append to it
        if os.path.exists(args.output):
            existing_df = pd.read_csv(args.output)
            combined_df = pd.concat([existing_df, new_summary_df], ignore_index=True)
            
            # Optional: Drop duplicates if you re-run the exact same file
            combined_df = combined_df.drop_duplicates(subset=['Method', 'Source_File'], keep='last')
        else:
            combined_df = new_summary_df

        combined_df.to_csv(args.output, index=False)
        print("\n" + "="*50)
        print(f"[SUCCESS] All individual averages combined and saved to: {args.output}")
        print("="*50)
    else:
        print("\n[ERROR] No valid data was processed.")

if __name__ == "__main__":
    main()


# python average_methods.py --inputs smd_results_mse.csv smd_results_interval.csv --output smd_final_comparison.csv