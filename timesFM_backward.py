# BACKWARD:-
#
# Symmetric counterpart to timesFM_modularised.py (FORWARD).
# For each window [s, s+h), use the FUTURE rows [s+h, s+h+context_length)
# REVERSED as context, ask TimesFM to predict h steps in fake-time, then flip
# the quantile predictions back so row i aligns with real-time index s+i.
#
# When the future runs out near the end of the series, fall back to a forward
# forecast from the preceding past so every row of df still gets a score.

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

# TSB-AD Imports
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- CONFIG --------------------
SEED              = 2024
WINDOWS_PER_BATCH = 10
SMOOTH_WINDOW     = 5

# TimesFM 2.5 hard maximums (set at compile time).
MAX_CONTEXT = 1024
MAX_HORIZON = 256

# TimesFM quantile_forecast layout: (batch, horizon, 10) = [mean, q10, q20, ..., q90]
Q10_IDX, Q50_IDX, Q90_IDX = 1, 5, 9


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
    parser = argparse.ArgumentParser(description="TimesFM Zero-Shot Backward Anomaly Detection (TSB-AD)")
    parser.add_argument(
        "--data_pattern", type=str, default="./mTSBench/SMD/*test.csv",
        help="Glob pattern for input test CSVs",
    )
    parser.add_argument(
        "--save_path", type=str, default="smd_timesfm_backward_results.csv",
        help="Where to save per-file metrics CSV",
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help=f"Number of FUTURE timesteps used as (reversed) model context (max {MAX_CONTEXT})",
    )
    parser.add_argument(
        "--horizon", type=int, default=128,
        help=f"Number of timesteps scored per window (max {MAX_HORIZON})",
    )
    parser.add_argument(
        "--windows_per_batch", type=int, default=WINDOWS_PER_BATCH,
        help="Number of forecast windows batched into one model.forecast call",
    )
    parser.add_argument(
        "--score_method", type=str, default="interval",
        choices=["mse", "interval", "normalized_deviation", "smape"],
        help=(
            "Anomaly scoring method per feature:\n"
            "  mse                 - squared error vs median\n"
            "  interval            - violation beyond [0.1, 0.9] quantile band\n"
            "  normalized_deviation- |actual - median| / band_width\n"
            "  smape               - symmetric MAPE vs median"
        ),
    )
    parser.add_argument(
        "--agg_method", type=str, default="l2",
        choices=["l2", "max", "mean", "topk_mean"],
        help=(
            "How to aggregate per-feature scores into a single time-series score:\n"
            "  l2        - L2 norm\n"
            "  max       - maximum across features\n"
            "  mean      - mean across features\n"
            "  topk_mean - mean of top-k features"
        ),
    )
    return parser.parse_args()


# -------------------- MODEL --------------------
def load_model(weights_path="./timesfm-weights"):
    """Load TimesFM 2.5 and compile with our inference config."""
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
    """Read a CSV and return (df, feature_list, label, n_total, n_dim)."""
    df = pd.read_csv(file_path).dropna()
    label_col    = "Label" if "Label" in df.columns else "is_anomaly"
    feature_list = [c for c in df.columns if c != label_col and c != "timestamp"]

    data         = df[feature_list].values.astype(float)
    label        = df[label_col].astype(int).to_numpy()
    n_total, n_dim = data.shape

    return df, feature_list, label, n_total, n_dim


def compute_sliding_window(data, n_dim):
    """Adaptive VUS sliding window from the dominant ACF period."""
    if n_dim == 1:
        return find_length_rank(data, rank=1)
    return find_length_rank(data[:, 0].reshape(-1, 1), rank=1)


# -------------------- PREDICTION --------------------
def generate_backward_prediction(model, df, feature_list, n_total,
                                 context_length, horizon, windows_per_batch):
    """Run batched TimesFM forecasts using REVERSED FUTURE rows as context.

    For each window starting at `start` (step = horizon, walking 0 → n_total):
      - Real-time target  window is [start, start + h).
      - Real-time future context is [start + h, start + h + context_length),
        capped at n_total.
      - If at least `h` future rows are available, reverse them and ask TimesFM
        to predict `horizon` steps in fake-time. The first predicted step lines
        up (after flipping) with real-time index `start + h - 1`, and the last
        with index `start`.
      - If the future is too short (final window region), FALL BACK to a
        forward forecast from the preceding past so every row of df is scored.

    Returns a long-format DataFrame with columns:
        target_name, t_idx, 0.1, 0.5, 0.9
    where t_idx is the absolute timestep index in the original series.
    Coverage is the full range [0, n_total) thanks to the fallback.
    """
    starts = list(range(0, n_total, horizon))
    n_dim  = len(feature_list)

    # Each list holds slices of shape (n_dim, h_i) for each window i.
    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []
    fallback_count = 0

    pbar = tqdm(
        total=len(starts),
        desc="  backward windows",
        unit="win",
        leave=False,
        position=1,
    )

    for i in range(0, len(starts), windows_per_batch):
        batch_starts   = starts[i : i + windows_per_batch]
        batch_inputs   = []
        valid_horizons = []
        flip_flags     = []  # True if window is backward (must flip predictions)

        # Build inputs: for each window, append n_dim univariate context arrays.
        for start in batch_starts:
            h = min(horizon, n_total - start)
            valid_horizons.append(h)

            fut_start     = start + h
            fut_end       = min(fut_start + context_length, n_total)
            fut_available = fut_end - fut_start

            if fut_available >= h:
                # Backward path: reverse the future rows.
                ctx_df = df.iloc[fut_start:fut_end]
                flip_flags.append(True)
                for f in feature_list:
                    arr = ctx_df[f].to_numpy(dtype=np.float32)
                    batch_inputs.append(arr[::-1].copy())  # contiguous reversed
            else:
                # Fallback: forward forecast from preceding past.
                past_start = max(0, start - context_length)
                ctx_df     = df.iloc[past_start:start]
                if len(ctx_df) == 0:
                    # Degenerate case: no future AND no past. Skip this window.
                    # (Only possible on absurdly short series; outer caller
                    # already gates on n_total > context_length.)
                    valid_horizons[-1] = 0
                    flip_flags.append(False)
                    for _ in feature_list:
                        batch_inputs.append(np.zeros(1, dtype=np.float32))
                    continue
                flip_flags.append(False)
                fallback_count += 1
                for f in feature_list:
                    batch_inputs.append(ctx_df[f].to_numpy(dtype=np.float32))

        # One model call for the whole batch.
        _, quantile_forecast = model.forecast(horizon=horizon, inputs=batch_inputs)

        # Slice back into per-window groups of n_dim rows; flip if backward.
        current_idx = 0
        for h, start, flip in zip(valid_horizons, batch_starts, flip_flags):
            window_forecast = quantile_forecast[current_idx : current_idx + n_dim]
            current_idx    += n_dim
            pbar.update(1)
            if h == 0:
                continue  # degenerate skip

            q10 = window_forecast[:, :h, Q10_IDX]
            q50 = window_forecast[:, :h, Q50_IDX]
            q90 = window_forecast[:, :h, Q90_IDX]
            if flip:
                # Reversed-context predictions come out in reverse real-time
                # order. Flip along the time axis so row i aligns with start+i.
                q10 = q10[:, ::-1]
                q50 = q50[:, ::-1]
                q90 = q90[:, ::-1]
            q10_chunks.append(q10)
            q50_chunks.append(q50)
            q90_chunks.append(q90)
            t_chunks.append(np.arange(start, start + h))

    pbar.close()

    if fallback_count > 0:
        print(f"  fallback (forward-from-past) windows: {fallback_count}")

    # Concatenate along time axis -> (n_dim, T_forecasted)
    q10 = np.concatenate(q10_chunks, axis=1)
    q50 = np.concatenate(q50_chunks, axis=1)
    q90 = np.concatenate(q90_chunks, axis=1)
    t   = np.concatenate(t_chunks)

    # Build long-format DataFrame matching the forward script's output.
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


# -------------------- ANOMALY SCORING (shared with forward) --------------------
def compute_feature_score(y_actual, group_df, method="interval"):
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


def aggregate_scores(anomaly_df, method="l2"):
    if method == "l2":
        return np.sqrt((anomaly_df ** 2).sum(axis=1)).values

    elif method == "max":
        return anomaly_df.max(axis=1).values

    elif method == "mean":
        return anomaly_df.mean(axis=1).values

    else:  # topk_mean
        k = 4
        return anomaly_df.apply(
            lambda row: row.nlargest(k).mean(), axis=1
        ).values


def robust_normalize(series):
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


def compute_anomaly_score(prediction_df, df, feature_list, score_method, agg_method):
    """Per-feature score → robust normalize (skipped for sMAPE) → aggregate."""
    anomaly_scores = {}
    for feature_name, group_df in prediction_df.groupby("target_name"):
        group_df = group_df.sort_values("t_idx").reset_index(drop=True)

        # Align actual values to the forecasted timestep range.
        t_start = int(group_df["t_idx"].iloc[0])
        t_end   = int(group_df["t_idx"].iloc[-1]) + 1
        y_actual = df[feature_name].iloc[t_start:t_end].to_numpy(dtype=np.float32)

        anomaly_scores[feature_name] = compute_feature_score(
            y_actual, group_df, method=score_method
        )

    anomaly_df = pd.DataFrame(anomaly_scores)

    if score_method != "smape":
        anomaly_df = anomaly_df.apply(
            lambda col: pd.Series(robust_normalize(col.values)), axis=0
        )
    anomaly_df = anomaly_df.fillna(0)

    return aggregate_scores(anomaly_df, method=agg_method)


# -------------------- POST-PROCESSING --------------------
def smooth_score(y_score, smooth_window):
    """Backward (with fallback) already covers [0, n_total), so no padding is
    needed — just smooth."""
    if smooth_window > 1:
        return uniform_filter1d(y_score, size=smooth_window)
    return y_score


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, file_path, score_method, agg_method,
                 context_length, horizon, windows_per_batch):
    """Run the full pipeline on one file. Returns metric dict or None if skipped."""
    df, feature_list, label, n_total, n_dim = load_and_prepare(file_path)

    if label.sum() == 0:
        return None

    # Skip files too short to provide one full context window.
    if n_total <= context_length:
        print(f"  Skipping (n_total={n_total} <= context_length={context_length})")
        return None

    data = df[feature_list].values.astype(float)
    slidingWindow = compute_sliding_window(data, n_dim)

    prediction_df = generate_backward_prediction(
        model, df, feature_list, n_total,
        context_length, horizon, windows_per_batch,
    )

    y_score = compute_anomaly_score(
        prediction_df, df, feature_list, score_method, agg_method
    )

    y_score = smooth_score(y_score, SMOOTH_WINDOW)

    result = get_metrics(y_score, label, slidingWindow=slidingWindow)

    return {
        "AUROC":   result["AUC-ROC"],
        "AUPRC":   result["AUC-PR"],
        "VUS-ROC": result["VUS-ROC"],
        "VUS-PR":  result["VUS-PR"],
    }


# -------------------- SUMMARY --------------------
def summarize_and_save(results, save_path, score_method):
    if not results["file_name"]:
        print("No files were successfully processed.")
        return

    pd.DataFrame(results).to_csv(save_path, index=False)

    avg_auroc   = np.mean(results["AUROC"])
    avg_auprc   = np.mean(results["AUPRC"])
    avg_vus_roc = np.mean(results["VUS-ROC"])
    avg_vus_pr  = np.mean(results["VUS-PR"])

    print("\n" + "=" * 50)
    print(f"FINAL AVERAGE METRICS — BACKWARD (Method: {score_method.upper()})")
    print("=" * 50)
    print(f"Mean AUROC:   {avg_auroc:.4f}")
    print(f"Mean AUPRC:   {avg_auprc:.4f}")
    print(f"Mean VUS-ROC: {avg_vus_roc:.4f}")
    print(f"Mean VUS-PR:  {avg_vus_pr:.4f}")
    print("=" * 50)
    print(f"\n[SUCCESS] Per-file results saved to: {save_path}")


# -------------------- MAIN --------------------
def main():
    set_seed(SEED)
    print("CUDA Available:", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())

    args = parse_args()

    # Validate against compile-time caps.
    if args.context_length > MAX_CONTEXT:
        raise ValueError(
            f"--context_length {args.context_length} exceeds MAX_CONTEXT={MAX_CONTEXT}. "
            f"Raise MAX_CONTEXT and recompile if you need a longer context."
        )
    if args.horizon > MAX_HORIZON:
        raise ValueError(
            f"--horizon {args.horizon} exceeds MAX_HORIZON={MAX_HORIZON}. "
            f"Raise MAX_HORIZON and recompile if you need a longer horizon."
        )

    print(
        f"Config: direction=BACKWARD score={args.score_method} agg={args.agg_method} "
        f"context={args.context_length} horizon={args.horizon} "
        f"windows_per_batch={args.windows_per_batch}"
    )

    print("Loading TimesFM...")
    model = load_model()

    file_list = glob.glob(args.data_pattern)
    if not file_list:
        print(f"No files found matching pattern: {args.data_pattern}")
        return
    print(f"Found {len(file_list)} files to process.")

    results = defaultdict(list)

    for file_path in tqdm(
        file_list,
        desc=f"Processing Files [BACKWARD] ({args.score_method})",
        position=0,
    ):
        file_name = os.path.basename(file_path).replace(".csv", "")

        metrics = process_file(
            model, file_path,
            args.score_method, args.agg_method,
            args.context_length, args.horizon, args.windows_per_batch,
        )
        if metrics is None:
            continue

        results["file_name"].append(file_name)
        for k, v in metrics.items():
            results[k].append(v)

    summarize_and_save(results, args.save_path, args.score_method)


if __name__ == "__main__":
    main()


# Default run:
# python timesFM_backward.py --score_method interval --save_path smd_backward_interval.csv

# Ablation examples:
# python timesFM_backward.py --context_length 256  --horizon 32  --save_path smd_bwd_c256_h32.csv
# python timesFM_backward.py --context_length 1024 --horizon 100 --save_path smd_bwd_c1024_h100.csv