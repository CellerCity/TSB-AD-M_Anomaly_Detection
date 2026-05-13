
import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import timesfm
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

# TSB-AD Imports
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

# -------------------- SEEDING --------------------
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

# -------------------- TIMESFM CONFIG --------------------
CONTEXT_LENGTH = 512
HORIZON        = 128
WINDOWS_PER_BATCH = 10 

if __name__ == '__main__':
    # -------------------- ARGUMENTS --------------------
    parser = argparse.ArgumentParser(description='Running TSB-AD with TimesFM Zero-Shot')
    parser.add_argument('--data_pattern', type=str, default='./mTSBench/SMD/*test.csv')
    parser.add_argument('--save_path', type=str, default='smd_timesfm_results.csv')
    
    # NEW ARGUMENT: Choose the scoring method
    parser.add_argument('--score_method', type=str, default='interval', 
                        choices=['interval', 'mse'], 
                        help="Choose 'interval' (tunneling) or 'mse' (point forecast error)")
    args = parser.parse_args()

    # -------------------- LOAD TIMESFM MODEL --------------------
    print(f"Loading TimesFM... (Scoring Method: {args.score_method.upper()})")
    # model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    # Point directly to the local folder 
    model = TimesFM_2p5_200M_torch.from_pretrained("./timesfm-weights")

    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=256
        )
    )

    # -------------------- PREPARE FILE LOOP --------------------
    file_list = glob.glob(args.data_pattern)
    if not file_list:
        print(f"No files found matching pattern: {args.data_pattern}")
        exit()

    print(f"Found {len(file_list)} files to process.")
    dic_for_each_file = defaultdict(list)

    # -------------------- MAIN PROCESSING LOOP --------------------
    for file_path in tqdm(file_list, desc=f"Processing Files ({args.score_method})", position=0):
        file_name = os.path.basename(file_path).replace(".csv", "")
        
        # 1. Load Data
        df = pd.read_csv(file_path).dropna()
        label_col = 'Label' if 'Label' in df.columns else 'is_anomaly'
        
        feature_list = [c for c in df.columns if c != label_col and c != 'timestamp']
        data = df[feature_list].values.astype(float)
        label = df[label_col].astype(int).to_numpy()
        n_total, n_dim = data.shape

        if label.sum() == 0:
            continue

        # 2. Sliding Window Calculation
        if n_dim == 1:
            slidingWindow = find_length_rank(data, rank=1)
        else:
            slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

        # 3. Fast Batched Forecasting
        starts = list(range(CONTEXT_LENGTH, n_total, HORIZON))
        q10_all, q50_all, q90_all = [], [], []

        for i in range(0, len(starts), WINDOWS_PER_BATCH):
            batch_starts = starts[i : i + WINDOWS_PER_BATCH]
            batch_inputs, valid_horizons = [], []
            
            for start in batch_starts:
                h = min(HORIZON, n_total - start)
                valid_horizons.append(h)
                
                context_df = df.iloc[start - CONTEXT_LENGTH : start]
                for f in feature_list:
                    batch_inputs.append(context_df[f].to_numpy(dtype=np.float32))

            _, quantile_forecast = model.forecast(horizon=HORIZON, inputs=batch_inputs)
            
            current_idx = 0
            for h in valid_horizons:
                window_forecast = quantile_forecast[current_idx : current_idx + n_dim]
                q10_all.append(window_forecast[:, :h, 1])
                q50_all.append(window_forecast[:, :h, 5]) # Extract Median (q50)
                q90_all.append(window_forecast[:, :h, 9])
                current_idx += n_dim

        q10 = np.concatenate(q10_all, axis=1)
        q50 = np.concatenate(q50_all, axis=1)
        q90 = np.concatenate(q90_all, axis=1)

        # 4. Anomaly Score Calculation
        y_actual = df[feature_list].iloc[CONTEXT_LENGTH:].to_numpy(dtype=np.float32).T  
        
        # --- SCORING BRANCH ---
        if args.score_method == 'interval':
            per_feat = np.maximum(0.0, y_actual - q90) + np.maximum(0.0, q10 - y_actual)
        elif args.score_method == 'mse':
            per_feat = (y_actual - q50) ** 2
        # ----------------------

        p1  = np.percentile(per_feat,  1, axis=1, keepdims=True)
        p99 = np.percentile(per_feat, 99, axis=1, keepdims=True)
        per_feat = np.clip(per_feat, p1, p99)
        per_feat = np.where(p99 - p1 > 1e-8, (per_feat - p1) / (p99 - p1 + 1e-8), 0.0)

        y_score_forecasted = np.sqrt((per_feat ** 2).sum(axis=0))

        output = np.zeros(n_total)
        output[:CONTEXT_LENGTH] = y_score_forecasted[0]  ## Keeping the intial context-length forecast values as 
        output[CONTEXT_LENGTH:] = y_score_forecasted
        output = uniform_filter1d(output, size=5)

        # 5. TSB-AD Evaluation
        # output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
        # binary_preds = output > (np.mean(output) + 3 * np.std(output))
        
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=binary_preds)
        
        # 6. Store Results
        dic_for_each_file["file_name"].append(file_name)
        dic_for_each_file["AUROC"].append(evaluation_result["AUC-ROC"])
        dic_for_each_file["AUPRC"].append(evaluation_result["AUC-PR"])
        dic_for_each_file["VUS-ROC"].append(evaluation_result["VUS-ROC"])
        dic_for_each_file["VUS-PR"].append(evaluation_result["VUS-PR"])

    # -------------------- SUMMARY AND EXPORT --------------------
    if len(dic_for_each_file["file_name"]) > 0:
        # 1. Save per-file results
        results_df = pd.DataFrame(dic_for_each_file)
        results_df.to_csv(args.save_path, index=False)

        # 2. Calculate Averages
        avg_auroc = np.mean(dic_for_each_file['AUROC'])
        avg_auprc = np.mean(dic_for_each_file['AUPRC'])
        avg_vus_roc = np.mean(dic_for_each_file['VUS-ROC'])
        avg_vus_pr = np.mean(dic_for_each_file['VUS-PR'])

        # 3. Print Averages
        print("\n" + "="*50)
        print(f"FINAL AVERAGE METRICS (Method: {args.score_method.upper()})")
        print("="*50)
        print(f"Mean AUROC:   {avg_auroc:.4f}")
        print(f"Mean AUPRC:   {avg_auprc:.4f}")
        print(f"Mean VUS-ROC: {avg_vus_roc:.4f}")
        print(f"Mean VUS-PR:  {avg_vus_pr:.4f}")
        print("="*50)

        print(f"\n[SUCCESS] Per-file results saved to: {args.save_path}")





# python timesFM_sample.py --score_method mse --save_path smd_results_mse.csv

# python timesFM_sample.py --score_method interval --save_path smd_results_interval.csv