# BIDIRECTIONAL:-
#
# Combines FORWARD (timesFM_modularised.py style) and BACKWARD (pure, no
# fallback) predictions, then fuses the per-feature anomaly scores using one of
# {max, mean, min, fwd_only, bwd_only} before aggregating across features.
#
# Coverage in TimesFM (no-train-split) setup:
#   - Forward  : real forecasts on [context_length, n_total).  First
#                context_length rows have no forward prediction → NaN.
#   - Backward : real forecasts on [0, n_backward_rows).  We stop the backward
#                walk the first time the remaining future is shorter than a full
#                window (mirrors the senior's bidirectional.py logic). Trailing
#                rows have no backward prediction → NaN.
#
# Fusion is NaN-aware (np.fmax / np.nanmean / np.fmin), so wherever one
# direction is missing the other one carries the score for that timestep.

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
    parser = argparse.ArgumentParser(description="TimesFM Zero-Shot Bidirectional Anomaly Detection (TSB-AD)")
    parser.add_argument(
        "--data_pattern", type=str, default="./mTSBench/SMD/*test.csv",
        help="Glob pattern for input test CSVs",
    )
    parser.add_argument(
        "--save_path", type=str, default="smd_timesfm_bidirectional_results.csv",
        help="Where to save per-file metrics CSV",
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help=f"Number of past/future timesteps used as model context (max {MAX_CONTEXT})",
    )
    parser.add_argument(
        "--horizon", type=int, default=128,
        help=f"Number of future timesteps forecasted per window (max {MAX_HORIZON})",
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
    parser.add_argument(
        "--fusion_method", type=str, default="max",
        choices=["max", "mean", "min", "fwd_only", "bwd_only"],
        help=(
            "How to fuse forward and backward per-feature anomaly scores:\n"
            "  max      - elementwise max (anomaly fires if either direction is surprised)\n"
            "  mean     - elementwise mean\n"
            "  min      - elementwise min (anomaly only if both directions agree)\n"
            "  fwd_only - ignore backward (ablation)\n"
            "  bwd_only - ignore forward (ablation)"
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


# -------------------- FORWARD PREDICTION --------------------
def generate_forward_prediction(model, df, feature_list, n_total,
                                context_length, horizon, windows_per_batch):
    """Batched TimesFM forward forecasts walking from context_length to n_total.

    Returns a long-format DataFrame with columns:
        target_name, t_idx, 0.1, 0.5, 0.9
    Coverage: t_idx in [context_length, n_total).
    """
    starts = list(range(context_length, n_total, horizon))
    n_dim  = len(feature_list)

    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []

    pbar = tqdm(
        total=len(starts),
        desc="  forward windows",
        unit="win",
        leave=False,
        position=1,
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


# -------------------- BACKWARD PREDICTION (PURE, NO FALLBACK) --------------------
def generate_backward_prediction(model, df, feature_list, n_total,
                                 context_length, horizon, windows_per_batch):
    """Batched TimesFM backward forecasts using reversed future as context.

    Pure backward (no forward fallback) — when a window does not have at least
    `h` rows of future remaining, we STOP. This mirrors the senior's
    bidirectional.py: backward gives a contiguous prefix [0, n_backward_rows),
    fusion handles the tail.

    Returns (prediction_df, n_backward_rows). prediction_df may be None if no
    window has enough future (only possible on extremely short series).
    """
    n_dim   = len(feature_list)
    starts  = list(range(0, n_total, horizon))

    # First, decide which window starts have sufficient future. We stop at the
    # first one that doesn't (later ones can only be worse).
    backward_starts   = []
    backward_horizons = []
    for start in starts:
        h = min(horizon, n_total - start)
        fut_start     = start + h
        fut_end       = min(fut_start + context_length, n_total)
        fut_available = fut_end - fut_start
        if fut_available < h:
            break
        backward_starts.append(start)
        backward_horizons.append(h)

    n_backward_rows = sum(backward_horizons)
    if not backward_starts:
        return None, 0

    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []

    pbar = tqdm(
        total=len(backward_starts),
        desc="  backward windows",
        unit="win",
        leave=False,
        position=1,
    )

    for i in range(0, len(backward_starts), windows_per_batch):
        batch_starts   = backward_starts[i : i + windows_per_batch]
        batch_h        = backward_horizons[i : i + windows_per_batch]
        batch_inputs   = []

        for start, h in zip(batch_starts, batch_h):
            fut_start = start + h
            fut_end   = min(fut_start + context_length, n_total)
            ctx_df    = df.iloc[fut_start:fut_end]
            for f in feature_list:
                arr = ctx_df[f].to_numpy(dtype=np.float32)
                batch_inputs.append(arr[::-1].copy())  # contiguous reversed

        _, quantile_forecast = model.forecast(horizon=horizon, inputs=batch_inputs)

        current_idx = 0
        for h, start in zip(batch_h, batch_starts):
            window_forecast = quantile_forecast[current_idx : current_idx + n_dim]
            current_idx    += n_dim
            pbar.update(1)

            # Flip along the time axis: reversed-context predictions come out
            # in reverse real-time order.
            q10 = window_forecast[:, :h, Q10_IDX][:, ::-1]
            q50 = window_forecast[:, :h, Q50_IDX][:, ::-1]
            q90 = window_forecast[:, :h, Q90_IDX][:, ::-1]
            q10_chunks.append(q10)
            q50_chunks.append(q50)
            q90_chunks.append(q90)
            t_chunks.append(np.arange(start, start + h))

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
    return pd.concat(frames, ignore_index=True), n_backward_rows


# -------------------- ANOMALY SCORING --------------------
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


def build_full_length_score_df(prediction_df, df, feature_list, n_total, score_method):
    """Return a DataFrame of shape (n_total, n_features) with NaN where this
    direction has no prediction. Assumes t_idx within each feature group is
    contiguous (which holds for both our forward and backward walks).
    """
    score_columns = {}
    for feature_name in feature_list:
        score_arr = np.full(n_total, np.nan, dtype=float)

        if prediction_df is not None and not prediction_df.empty:
            grp = prediction_df[prediction_df["target_name"] == feature_name]
            grp = grp.sort_values("t_idx").reset_index(drop=True)
            if len(grp) > 0:
                t_start  = int(grp["t_idx"].iloc[0])
                t_end    = int(grp["t_idx"].iloc[-1]) + 1
                y_actual = df[feature_name].iloc[t_start:t_end].to_numpy(dtype=np.float32)
                scores   = compute_feature_score(y_actual, grp, method=score_method)
                score_arr[t_start:t_end] = scores

        score_columns[feature_name] = score_arr

    return pd.DataFrame(score_columns)


def normalize_nansafe(anomaly_df):
    """Robust-normalize each column independently, leaving NaN entries as NaN."""
    def _normalize_col(col):
        arr  = col.values.astype(float)
        mask = ~np.isnan(arr)
        out  = np.full_like(arr, np.nan, dtype=float)
        if mask.any():
            out[mask] = robust_normalize(arr[mask])
        return pd.Series(out, index=col.index)
    return anomaly_df.apply(_normalize_col, axis=0)


def fuse_anomaly_dfs(fwd_df, bwd_df, method):
    """Elementwise NaN-aware fusion of forward and backward per-feature scores.

    Both inputs have shape (n_total, n_features). Rows beyond a direction's
    coverage are NaN. Fusion ops:
      - max  : np.fmax  (NaN-safe; falls back to the other direction where one is NaN)
      - mean : np.nanmean across the two directions
      - min  : np.fmin
      - fwd_only / bwd_only : ablations
    For bwd_only, rows without backward coverage fall back to forward so every
    test row still has a score.
    """
    if method == "fwd_only":
        return fwd_df.copy()
    if method == "bwd_only":
        out     = bwd_df.copy()
        missing = out.isna().any(axis=1)
        out.loc[missing] = fwd_df.loc[missing]
        return out

    fwd_arr = fwd_df.values
    bwd_arr = bwd_df.values

    if method == "max":
        fused = np.fmax(fwd_arr, bwd_arr)
    elif method == "mean":
        fused = np.nanmean(np.stack([fwd_arr, bwd_arr], axis=0), axis=0)
    else:  # min
        fused = np.fmin(fwd_arr, bwd_arr)

    return pd.DataFrame(fused, columns=fwd_df.columns)


def compute_bidirectional_score(fwd_pred_df, bwd_pred_df, df, feature_list,
                                n_total, score_method, agg_method, fusion_method):
    """Build per-direction full-length score DataFrames, normalize each
    independently, fuse, fill NaN, and aggregate across features."""
    fwd_anomaly_df = build_full_length_score_df(
        fwd_pred_df, df, feature_list, n_total, score_method
    )
    bwd_anomaly_df = build_full_length_score_df(
        bwd_pred_df, df, feature_list, n_total, score_method
    )

    # Normalize each direction independently so fusion is on the same scale.
    if score_method != "smape":
        fwd_anomaly_df = normalize_nansafe(fwd_anomaly_df)
        bwd_anomaly_df = normalize_nansafe(bwd_anomaly_df)

    anomaly_df = fuse_anomaly_dfs(fwd_anomaly_df, bwd_anomaly_df, fusion_method)
    anomaly_df = anomaly_df.fillna(0)

    return aggregate_scores(anomaly_df, method=agg_method)


# -------------------- POST-PROCESSING --------------------
def smooth_score(y_score, smooth_window):
    """y_score is already length n_total (NaN entries were filled before
    aggregation), so we only smooth here."""
    if smooth_window > 1:
        return uniform_filter1d(y_score, size=smooth_window)
    return y_score


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, file_path, score_method, agg_method, fusion_method,
                 context_length, horizon, windows_per_batch):
    """Run the full pipeline on one file. Returns metric dict or None if skipped."""
    df, feature_list, label, n_total, n_dim = load_and_prepare(file_path)

    if label.sum() == 0:
        return None

    if n_total <= context_length:
        print(f"  Skipping (n_total={n_total} <= context_length={context_length})")
        return None

    data = df[feature_list].values.astype(float)
    slidingWindow = compute_sliding_window(data, n_dim)

    # --- forward + backward predictions ---
    fwd_pred_df = generate_forward_prediction(
        model, df, feature_list, n_total,
        context_length, horizon, windows_per_batch,
    )
    bwd_pred_df, n_backward_rows = generate_backward_prediction(
        model, df, feature_list, n_total,
        context_length, horizon, windows_per_batch,
    )

    print(f"  backward coverage: {n_backward_rows}/{n_total} rows "
          f"({(n_backward_rows / n_total * 100 if n_total else 0):.1f}%)")

    # --- fuse and aggregate ---
    y_score = compute_bidirectional_score(
        fwd_pred_df, bwd_pred_df, df, feature_list, n_total,
        score_method, agg_method, fusion_method,
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
def summarize_and_save(results, save_path, score_method, fusion_method):
    if not results["file_name"]:
        print("No files were successfully processed.")
        return

    pd.DataFrame(results).to_csv(save_path, index=False)

    avg_auroc   = np.mean(results["AUROC"])
    avg_auprc   = np.mean(results["AUPRC"])
    avg_vus_roc = np.mean(results["VUS-ROC"])
    avg_vus_pr  = np.mean(results["VUS-PR"])

    print("\n" + "=" * 50)
    print(
        f"FINAL AVERAGE METRICS — BIDIRECTIONAL "
        f"(score={score_method.upper()}, fusion={fusion_method.upper()})"
    )
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
        f"Config: direction=BIDIRECTIONAL score={args.score_method} "
        f"agg={args.agg_method} fusion={args.fusion_method} "
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
        desc=f"Processing Files [BIDIRECTIONAL] ({args.score_method}/{args.fusion_method})",
        position=0,
    ):
        file_name = os.path.basename(file_path).replace(".csv", "")

        metrics = process_file(
            model, file_path,
            args.score_method, args.agg_method, args.fusion_method,
            args.context_length, args.horizon, args.windows_per_batch,
        )
        if metrics is None:
            continue

        results["file_name"].append(file_name)
        for k, v in metrics.items():
            results[k].append(v)

    summarize_and_save(results, args.save_path, args.score_method, args.fusion_method)


if __name__ == "__main__":
    main()


# Default run:
# python timesFM_bidirectional.py --score_method interval --fusion_method max --save_path smd_bidir_interval_max.csv

# Fusion ablations:
# python timesFM_bidirectional.py --fusion_method mean     --save_path smd_bidir_mean.csv
# python timesFM_bidirectional.py --fusion_method min      --save_path smd_bidir_min.csv
# python timesFM_bidirectional.py --fusion_method fwd_only --save_path smd_bidir_fwd_only.csv
# python timesFM_bidirectional.py --fusion_method bwd_only --save_path smd_bidir_bwd_only.csv

# Context / horizon ablations:
# python timesFM_bidirectional.py --context_length 256  --horizon 32  --save_path smd_bidir_c256_h32.csv
# python timesFM_bidirectional.py --context_length 1024 --horizon 100 --save_path smd_bidir_c1024_h100.csv