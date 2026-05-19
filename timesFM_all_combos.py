# FORWARD - All scoring × aggregation combinations per (context, horizon) cell.
#
# Runs the TimesFM forecast ONCE per file, then evaluates all 16 combinations of:
#   score_method ∈ {mse, smape, interval, normalized_deviation}
#   agg_method   ∈ {l2, max, mean, topk_mean}
# Outputs a single CSV with one row per (file, score_method, agg_method).
#
# Metrics use fixed VUS parameters (no find_length_rank):
#   sliding_window=100, version='opt', thre=250

import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

import timesfm
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

from TSB_AD.evaluation.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- CONFIG --------------------
SEED              = 2024
WINDOWS_PER_BATCH = 10
SMOOTH_WINDOW     = 10

# TimesFM 2.5 hard maximums (set at compile time).
MAX_CONTEXT = 1024
MAX_HORIZON = 256

# TimesFM quantile_forecast layout: (batch, horizon, 10) = [mean, q10, q20, ..., q90]
Q10_IDX, Q50_IDX, Q90_IDX = 1, 5, 9

# Fixed VUS parameters for all evaluations (no adaptive find_length_rank).
VUS_SLIDING_WINDOW = 100
VUS_VERSION        = "opt"
VUS_THRE           = 250

# The grid we evaluate after each forecast.
SCORE_METHODS = ["mse", "smape", "interval", "normalized_deviation"]
AGG_METHODS   = ["l2", "max", "mean", "topk_mean"]
TOPK_K        = 4


# -------------------- REPRODUCIBILITY --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="TimesFM zero-shot AD: all score×agg combinations per forecast."
    )
    parser.add_argument(
        "--data_pattern", type=str, default="./mTSBench/SMD/*test.csv",
        help="Glob pattern for input test CSVs",
    )
    parser.add_argument(
        "--save_path", type=str, default="smd_timesfm_all_combos.csv",
        help="Where to save the all-combinations metrics CSV",
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help=f"Number of past timesteps used as model context (max {MAX_CONTEXT})",
    )
    parser.add_argument(
        "--horizon", type=int, default=128,
        help=f"Number of future timesteps forecasted per window (max {MAX_HORIZON})",
    )
    parser.add_argument(
        "--windows_per_batch", type=int, default=WINDOWS_PER_BATCH,
        help="Number of forecast windows batched into one model.forecast call",
    )
    return parser.parse_args()


# -------------------- MODEL --------------------
def load_model(weights_path="./timesfm-weights"):
    model = TimesFM_2p5_200M_torch.from_pretrained(weights_path)
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=256,
        )
    )
    return model


# -------------------- DATA LOADING --------------------
def load_and_prepare(file_path):
    df = pd.read_csv(file_path).dropna()
    label_col    = "Label" if "Label" in df.columns else "is_anomaly"
    feature_list = [c for c in df.columns if c != label_col and c != "timestamp"]

    data         = df[feature_list].values.astype(float)
    label        = df[label_col].astype(int).to_numpy()
    n_total, n_dim = data.shape

    return df, feature_list, label, n_total, n_dim


# -------------------- PREDICTION --------------------
def generate_prediction(model, df, feature_list, n_total, context_length, horizon, windows_per_batch):
    """Run batched TimesFM forecasts walking from context_length to n_total.

    Returns a long-format DataFrame with columns:
        target_name, t_idx, 0.1, 0.5, 0.9
    """
    starts = list(range(context_length, n_total, horizon))
    n_dim  = len(feature_list)

    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []

    pbar = tqdm(
        total=len(starts), desc="  forecasting windows",
        unit="win", leave=False, position=1,
    )

    for i in range(0, len(starts), windows_per_batch):
        batch_starts   = starts[i : i + windows_per_batch]
        batch_inputs   = []
        valid_horizons = []

        for start in batch_starts:
            h = min(horizon, n_total - start)
            valid_horizons.append(h)
            context_df = df.iloc[start - context_length : start]
            for f in feature_list:
                batch_inputs.append(context_df[f].to_numpy(dtype=np.float32))

        _, quantile_forecast = model.forecast(horizon=horizon, inputs=batch_inputs)

        current_idx = 0
        for h, start in zip(valid_horizons, batch_starts):
            window_forecast = quantile_forecast[current_idx : current_idx + n_dim]
            q10_chunks.append(window_forecast[:, :h, Q10_IDX])
            q50_chunks.append(window_forecast[:, :h, Q50_IDX])
            q90_chunks.append(window_forecast[:, :h, Q90_IDX])
            t_chunks.append(np.arange(start, start + h))
            current_idx += n_dim
            pbar.update(1)

    pbar.close()

    q10 = np.concatenate(q10_chunks, axis=1)
    q50 = np.concatenate(q50_chunks, axis=1)
    q90 = np.concatenate(q90_chunks, axis=1)
    t   = np.concatenate(t_chunks)

    frames = []
    for i, feat in enumerate(feature_list):
        frames.append(pd.DataFrame({
            "target_name": feat,
            "t_idx":       t,
            "0.1":         q10[i],
            "0.5":         q50[i],
            "0.9":         q90[i],
        }))
    return pd.concat(frames, ignore_index=True)


# -------------------- ANOMALY SCORING --------------------
def compute_feature_score(y_actual, group_df, method):
    y_median = group_df["0.5"].values
    y_lower  = group_df["0.1"].values
    y_upper  = group_df["0.9"].values

    if method == "mse":
        return (y_actual - y_median) ** 2

    elif method == "smape":
        eps = 1e-8
        return np.abs(y_actual - y_median) / (
            np.abs(y_actual) + np.abs(y_median) + eps
        )

    elif method == "interval":
        upper_violation = np.maximum(0.0, y_actual - y_upper)
        lower_violation = np.maximum(0.0, y_lower  - y_actual)
        return upper_violation + lower_violation

    else:  # normalized_deviation
        band_width = y_upper - y_lower + 1e-8
        deviation  = np.abs(y_actual - y_median)
        return deviation / band_width


def aggregate_scores(anomaly_df, method):
    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1)).values

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:  # topk_mean
        return anomaly_df.apply(
            lambda row: row.nlargest(TOPK_K).mean(), axis=1
        ).values


def robust_normalize(series):
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


# -------------------- POST-PROCESSING --------------------
def pad_and_smooth(y_score_forecasted, n_total, context_length, smooth_window):
    output = np.zeros(n_total)
    output[:context_length] = y_score_forecasted[0]
    output[context_length:] = y_score_forecasted
    if smooth_window > 1:
        output = uniform_filter1d(output, size=smooth_window)
    return output


# -------------------- ALL-COMBOS EVALUATION --------------------
def build_per_feature_score_df(prediction_df, df, feature_list, score_method):
    """Compute per-feature anomaly scores → robust normalize (unless sMAPE).
    Returns a DataFrame of shape (T_forecast, n_dim) with columns = feature names."""
    per_feature = {}
    for feature_name in feature_list:
        group_df = prediction_df[prediction_df["target_name"] == feature_name].sort_values("t_idx")
        t_start = int(group_df["t_idx"].iloc[0])
        t_end   = int(group_df["t_idx"].iloc[-1]) + 1
        y_actual = df[feature_name].iloc[t_start:t_end].to_numpy(dtype=np.float32)
        per_feature[feature_name] = compute_feature_score(y_actual, group_df, method=score_method)

    anomaly_df = pd.DataFrame(per_feature)

    if score_method != "smape":
        anomaly_df = anomaly_df.apply(
            lambda col: pd.Series(robust_normalize(col.values)), axis=0
        )
    return anomaly_df.fillna(0)


def evaluate_all_combos(prediction_df, df, feature_list, label, n_total, context_length):
    """For one forecast result, evaluate every (score_method, agg_method) combination.
    Returns a list of dicts (one per combination)."""
    if label.sum() == 0:
        return []  # caller skips files with no anomalies

    rows = []

    # Precompute the per-feature DataFrame once per score_method
    # (it's reused across all 4 aggregation methods for that score method).
    for score_method in SCORE_METHODS:
        anomaly_df = build_per_feature_score_df(prediction_df, df, feature_list, score_method)

        for agg_method in AGG_METHODS:
            y_score_forecasted = aggregate_scores(anomaly_df, method=agg_method)
            output = pad_and_smooth(
                y_score_forecasted, n_total, context_length, SMOOTH_WINDOW
            )

            result = get_metrics(
                output, label,
                slidingWindow=VUS_SLIDING_WINDOW,
                version=VUS_VERSION,
                thre=VUS_THRE,
            )

            rows.append({
                "score_method": score_method,
                "agg_method":   agg_method,
                "AUROC":        result["AUC-ROC"],
                "AUPRC":        result["AUC-PR"],
                "VUS-ROC":      result["VUS-ROC"],
                "VUS-PR":       result["VUS-PR"],
            })

    return rows


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, file_path, context_length, horizon, windows_per_batch):
    """Forecast once, evaluate all 16 combinations.
    Returns a list of metric rows (one per score×agg combination) or [] if skipped."""
    df, feature_list, label, n_total, n_dim = load_and_prepare(file_path)

    if label.sum() == 0:
        return []
    if n_total <= context_length:
        print(f"  Skipping (n_total={n_total} <= context_length={context_length})")
        return []

    prediction_df = generate_prediction(
        model, df, feature_list, n_total,
        context_length, horizon, windows_per_batch,
    )

    return evaluate_all_combos(
        prediction_df, df, feature_list, label, n_total, context_length
    )


# -------------------- SUMMARY --------------------
def summarize_and_save(results, save_path, context_length, horizon):
    if not results:
        print("No rows to save.")
        return

    df_out = pd.DataFrame(results)
    df_out.to_csv(save_path, index=False)

    print("\n" + "=" * 70)
    print(f"FINAL AVERAGE METRICS  (context={context_length}, horizon={horizon})")
    print("=" * 70)

    # Group by (score_method, agg_method), average across files.
    group = df_out.groupby(["score_method", "agg_method"]).agg({
        "AUROC":   "mean",
        "AUPRC":   "mean",
        "VUS-ROC": "mean",
        "VUS-PR":  "mean",
    }).reset_index()

    # Pretty-print as a table.
    print(group.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 70)
    print(f"\n[SUCCESS] Per-file × per-combination results saved to: {save_path}")


# -------------------- MAIN --------------------
def main():
    set_seed(SEED)
    print("CUDA Available:", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())

    args = parse_args()

    if args.context_length > MAX_CONTEXT:
        raise ValueError(
            f"--context_length {args.context_length} exceeds MAX_CONTEXT={MAX_CONTEXT}."
        )
    if args.horizon > MAX_HORIZON:
        raise ValueError(
            f"--horizon {args.horizon} exceeds MAX_HORIZON={MAX_HORIZON}."
        )

    print(
        f"Config: context={args.context_length} horizon={args.horizon} "
        f"windows_per_batch={args.windows_per_batch}"
    )
    print(f"Will evaluate {len(SCORE_METHODS)} × {len(AGG_METHODS)} = "
          f"{len(SCORE_METHODS) * len(AGG_METHODS)} score×agg combinations per file.")

    print("Loading TimesFM...")
    model = load_model()

    file_list = glob.glob(args.data_pattern)
    if not file_list:
        print(f"No files found matching pattern: {args.data_pattern}")
        return
    print(f"Found {len(file_list)} files to process.")

    all_results = []

    for file_path in tqdm(file_list, desc="Processing files", position=0):
        file_name = os.path.basename(file_path).replace(".csv", "")

        file_rows = process_file(
            model, file_path,
            args.context_length, args.horizon, args.windows_per_batch,
        )
        if not file_rows:
            continue

        for row in file_rows:
            row["file_name"]      = file_name
            row["context_length"] = args.context_length
            row["horizon"]        = args.horizon
            all_results.append(row)

    # Reorder columns for readability.
    if all_results:
        col_order = [
            "file_name", "context_length", "horizon",
            "score_method", "agg_method",
            "AUROC", "AUPRC", "VUS-ROC", "VUS-PR",
        ]
        df_final = pd.DataFrame(all_results)[col_order]
        df_final.to_csv(args.save_path, index=False)

        summarize_and_save(all_results, args.save_path, args.context_length, args.horizon)


if __name__ == "__main__":
    main()


# Sample calling:
# python timesFM_all_combos.py --context_length 512 --horizon 128 --save_path smd_c512_h128_all.csv
#
# Sample driver bash:
# HORIZONS=(5 10 30 50 100)
# CONTEXTS=(256 512 1024)
# for h in "${HORIZONS[@]}"; do
#   for c in "${CONTEXTS[@]}"; do
#     python timesFM_all_combos.py \
#       --context_length $c --horizon $h \
#       --save_path "ablation_results/smd_c${c}_h${h}_all.csv"
#   done
# done