# BIDIRECTIONAL (TOTO 2.0, native multivariate):-
#
# Combines FORWARD and BACKWARD (pure, no fallback) Toto predictions, then fuses
# the per-feature anomaly scores using one of {max, mean, min, fwd_only, bwd_only}
# before aggregating across features.
#
# Coverage (no-train-split setup):
#   - Forward  : real forecasts on [context_length, n_total). First context_length
#                rows have no forward prediction -> NaN.
#   - Backward : real forecasts on [0, n_backward_rows). The backward walk stops the
#                first time the remaining future is shorter than a full window.
#                Trailing rows have no backward prediction -> NaN.
#
# Fusion is NaN-aware (np.fmax / np.nanmean / np.fmin), so wherever one direction
# is missing the other carries the score for that timestep.
#
# Toto difference vs TimesFM: each window is fed as ONE multivariate tensor
# (batch, n_var, time) with series_ids all 0 (full cross-variate attention).
# The backward context is reversed jointly across channels.
#
# Metrics use fixed VUS parameters: sliding_window=100, version='opt', thre=250

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

from toto2 import Toto2Model

from TSB_AD.evaluation.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- CONFIG --------------------
SEED              = 2024
WINDOWS_PER_BATCH = 10
SMOOTH_WINDOW     = 5

MODEL_SIZE = "4m"   # 4m | 22m | 313m | 1B | 2.5B

# Toto forecast() returns (Q=9, batch, n_var, horizon); knots [0.1..0.9], median at 4.
Q10_IDX, Q50_IDX, Q90_IDX = 0, 4, 8

HAS_MISSING_VALUES = True   # set False on Ampere+ GPU for flash-attn speedup

VUS_SLIDING_WINDOW = 100
VUS_VERSION        = "opt"
VUS_THRE           = 250


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
    parser = argparse.ArgumentParser(description="Toto 2.0 Zero-Shot Bidirectional Anomaly Detection (TSB-AD)")
    parser.add_argument("--data_pattern", type=str, default="./mTSBench/GHL/*test.csv",
                        help="Glob pattern for input test CSVs")
    parser.add_argument("--save_path", type=str, default="ghl_toto_bidirectional_results.csv",
                        help="Where to save per-file metrics CSV")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Past/future timesteps used as context (must be divisible by patch_size)")
    parser.add_argument("--horizon", type=int, default=128,
                        help="Number of future timesteps forecasted per window")
    parser.add_argument("--windows_per_batch", type=int, default=WINDOWS_PER_BATCH,
                        help="Number of forecast windows batched into one model.forecast call")
    parser.add_argument("--model_size", type=str, default=MODEL_SIZE,
                        help="Toto size: 4m | 22m | 313m | 1B | 2.5B")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    parser.add_argument("--score_method", type=str, default="interval",
                        choices=["mse", "interval", "normalized_deviation", "smape"],
                        help="Anomaly scoring method per feature")
    parser.add_argument("--agg_method", type=str, default="l2",
                        choices=["l2", "max", "mean", "topk_mean"],
                        help="How to aggregate per-feature scores into one series score")
    parser.add_argument("--fusion_method", type=str, default="max",
                        choices=["max", "mean", "min", "fwd_only", "bwd_only"],
                        help=(
                            "How to fuse forward and backward per-feature anomaly scores:\n"
                            "  max      - elementwise max (fires if either direction is surprised)\n"
                            "  mean     - elementwise mean\n"
                            "  min      - elementwise min (fires only if both agree)\n"
                            "  fwd_only - ignore backward (ablation)\n"
                            "  bwd_only - ignore forward (ablation)"
                        ))
    return parser.parse_args()


# -------------------- MODEL --------------------
def load_model(model_size, device):
    checkpoint = f"Datadog/Toto-2.0-{model_size}"
    model = Toto2Model.from_pretrained(checkpoint, map_location=device)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {checkpoint}: {n_params:,} params | patch_size={model.config.patch_size}")
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


# -------------------- TOTO FORECAST HELPER --------------------
def fit_context_to_patch(ctx, patch_size):
    """Snap a (n_var, real_len) context to a length divisible by patch_size.
    Front-crop to the largest multiple (keep boundary-adjacent timesteps) when
    real_len >= patch_size; otherwise front-pad to one patch and mask the pad.
    Returns (ctx2 (n_var, L), mask2 (n_var, L) bool), L a multiple of patch_size.
    """
    n_var, real_len = ctx.shape
    keep = (real_len // patch_size) * patch_size
    if keep >= patch_size:
        ctx2  = ctx[:, real_len - keep:].copy()
        mask2 = np.ones((n_var, keep), dtype=bool)
    else:
        pad   = patch_size - real_len
        ctx2  = np.concatenate([np.zeros((n_var, pad), dtype=ctx.dtype), ctx], axis=1)
        mask2 = np.concatenate([np.zeros((n_var, pad), dtype=bool),
                                np.ones((n_var, real_len), dtype=bool)], axis=1)
    return ctx2, mask2


def _toto_forecast_batch(model, context_stack, horizon, device, mask_stack=None):
    """context_stack: list of (n_var, ctx) arrays (uniform ctx). -> (9, B, n_var, horizon).
    mask_stack (optional) marks padded positions; when present has_missing_values=True.
    """
    target     = torch.from_numpy(np.stack(context_stack, axis=0)).to(device)
    series_ids = torch.zeros(target.shape[0], target.shape[1], dtype=torch.long, device=device)
    if mask_stack is None:
        target_mask = torch.ones_like(target, dtype=torch.bool)
        has_missing = HAS_MISSING_VALUES
    else:
        mask_np     = np.stack(mask_stack, axis=0)
        target_mask = torch.from_numpy(mask_np).to(device)
        has_missing = HAS_MISSING_VALUES or (not mask_np.all())
    quantiles = model.forecast(
        {"target": target, "target_mask": target_mask, "series_ids": series_ids},
        horizon=horizon, has_missing_values=has_missing,
    )
    return quantiles.detach().cpu().numpy()


# -------------------- FORWARD PREDICTION --------------------
def generate_forward_prediction(model, df, feature_list, n_total,
                                context_length, horizon, windows_per_batch, device):
    """Batched Toto multivariate forward forecasts walking from context_length to n_total.
    Coverage: t_idx in [context_length, n_total)."""
    starts = list(range(context_length, n_total, horizon))
    n_dim  = len(feature_list)
    feat_matrix = df[feature_list].to_numpy(dtype=np.float32)

    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []

    pbar = tqdm(total=len(starts), desc="  forward windows",
                unit="win", leave=False, position=1)

    for i in range(0, len(starts), windows_per_batch):
        batch_starts   = starts[i : i + windows_per_batch]
        context_stack  = []
        valid_horizons = []

        for start in batch_starts:
            h = min(horizon, n_total - start)
            valid_horizons.append(h)
            ctx = feat_matrix[start - context_length:start].T  # (n_dim, ctx)
            context_stack.append(np.ascontiguousarray(ctx))

        quant = _toto_forecast_batch(model, context_stack, horizon, device)  # (9,B,n_var,h)

        for b, (h, start) in enumerate(zip(valid_horizons, batch_starts)):
            q10_chunks.append(quant[Q10_IDX, b, :, :h])
            q50_chunks.append(quant[Q50_IDX, b, :, :h])
            q90_chunks.append(quant[Q90_IDX, b, :, :h])
            t_chunks.append(np.arange(start, start + h))
            pbar.update(1)

    pbar.close()

    q10 = np.concatenate(q10_chunks, axis=1)
    q50 = np.concatenate(q50_chunks, axis=1)
    q90 = np.concatenate(q90_chunks, axis=1)
    t   = np.concatenate(t_chunks)

    frames = []
    for i, feat in enumerate(feature_list):
        frames.append(pd.DataFrame({
            "target_name": feat, "t_idx": t,
            "0.1": q10[i], "0.5": q50[i], "0.9": q90[i],
        }))
    return pd.concat(frames, ignore_index=True)


# -------------------- BACKWARD PREDICTION (PURE, NO FALLBACK) --------------------
def generate_backward_prediction(model, df, feature_list, n_total,
                                 context_length, horizon, windows_per_batch, device):
    """Batched Toto backward forecasts using reversed future as context.

    Pure backward (no forward fallback): when a window lacks at least `h` rows of
    future remaining, STOP. Backward gives a contiguous prefix [0, n_backward_rows);
    fusion handles the tail.

    Returns (prediction_df, n_backward_rows). prediction_df may be None if no window
    has enough future (only on extremely short series).
    """
    n_dim   = len(feature_list)
    starts  = list(range(0, n_total, horizon))
    patch_size = model.config.patch_size
    feat_matrix = df[feature_list].to_numpy(dtype=np.float32)

    # Decide which window starts have sufficient future; stop at the first that doesn't.
    backward_starts, backward_horizons = [], []
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

    pbar = tqdm(total=len(backward_starts), desc="  backward windows",
                unit="win", leave=False, position=1)

    for i in range(0, len(backward_starts), windows_per_batch):
        batch_starts  = backward_starts[i : i + windows_per_batch]
        batch_h       = backward_horizons[i : i + windows_per_batch]
        context_stack = []
        mask_stack    = []

        for start, h in zip(batch_starts, batch_h):
            fut_start = start + h
            fut_end   = min(fut_start + context_length, n_total)
            # Reverse time jointly across channels, then transpose to (n_dim, ctx).
            ctx = feat_matrix[fut_start:fut_end][::-1, :].T.copy()
            ctx2, mask2 = fit_context_to_patch(ctx, patch_size)
            context_stack.append(ctx2)
            mask_stack.append(mask2)

        # Windows near the series tail crop to a smaller patch-multiple, so a batch
        # may be mixed-length; those run as singletons, uniform ones batch together.
        ctx_lens = [c.shape[1] for c in context_stack]
        if len(set(ctx_lens)) == 1:
            quant = _toto_forecast_batch(model, context_stack, horizon, device, mask_stack)
            per_window_quant = [quant[:, b] for b in range(quant.shape[1])]
        else:
            per_window_quant = [
                _toto_forecast_batch(model, [c], horizon, device, [m])[:, 0]
                for c, m in zip(context_stack, mask_stack)
            ]

        for wq, h, start in zip(per_window_quant, batch_h, batch_starts):
            pbar.update(1)
            # Flip along time: reversed-context predictions come out reverse-real-time.
            q10 = wq[Q10_IDX, :, :h][:, ::-1].copy()
            q50 = wq[Q50_IDX, :, :h][:, ::-1].copy()
            q90 = wq[Q90_IDX, :, :h][:, ::-1].copy()
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
            "target_name": feat, "t_idx": t,
            "0.1": q10[i], "0.5": q50[i], "0.9": q90[i],
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
        return np.abs(y_actual - y_median) / (np.abs(y_actual) + np.abs(y_median) + eps)
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
        return anomaly_df.apply(lambda row: row.nlargest(k).mean(), axis=1).values


def robust_normalize(series):
    p1  = np.percentile(series, 1)
    p99 = np.percentile(series, 99)
    clipped = np.clip(series, p1, p99)
    denom = p99 - p1
    if denom < 1e-8:
        return np.zeros_like(series, dtype=float)
    return (clipped - p1) / denom


def build_full_length_score_df(prediction_df, df, feature_list, n_total, score_method):
    """(n_total, n_features) DataFrame with NaN where this direction has no prediction.
    Assumes t_idx within each feature group is contiguous (holds for both walks)."""
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

    Both inputs are (n_total, n_features); rows beyond a direction's coverage are NaN.
      - max  : np.fmax  (NaN-safe; falls back to the other direction where one is NaN)
      - mean : np.nanmean across the two directions
      - min  : np.fmin
      - fwd_only / bwd_only : ablations
    For bwd_only, rows without backward coverage fall back to forward.
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
    """Build per-direction full-length score DataFrames, normalize each independently,
    fuse, fill NaN, and aggregate across features."""
    fwd_anomaly_df = build_full_length_score_df(fwd_pred_df, df, feature_list, n_total, score_method)
    bwd_anomaly_df = build_full_length_score_df(bwd_pred_df, df, feature_list, n_total, score_method)

    if score_method != "smape":
        fwd_anomaly_df = normalize_nansafe(fwd_anomaly_df)
        bwd_anomaly_df = normalize_nansafe(bwd_anomaly_df)

    anomaly_df = fuse_anomaly_dfs(fwd_anomaly_df, bwd_anomaly_df, fusion_method)
    anomaly_df = anomaly_df.fillna(0)

    return aggregate_scores(anomaly_df, method=agg_method)


# -------------------- POST-PROCESSING --------------------
def smooth_score(y_score, smooth_window):
    """y_score is already length n_total (NaN entries filled before aggregation); just smooth."""
    if smooth_window > 1:
        return uniform_filter1d(y_score, size=smooth_window)
    return y_score


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, file_path, score_method, agg_method, fusion_method,
                 context_length, horizon, windows_per_batch, device):
    df, feature_list, label, n_total, n_dim = load_and_prepare(file_path)

    if label.sum() == 0:
        return None
    if n_total <= context_length:
        print(f"  Skipping (n_total={n_total} <= context_length={context_length})")
        return None

    fwd_pred_df = generate_forward_prediction(
        model, df, feature_list, n_total, context_length, horizon, windows_per_batch, device,
    )
    bwd_pred_df, n_backward_rows = generate_backward_prediction(
        model, df, feature_list, n_total, context_length, horizon, windows_per_batch, device,
    )

    print(f"  backward coverage: {n_backward_rows}/{n_total} rows "
          f"({(n_backward_rows / n_total * 100 if n_total else 0):.1f}%)")

    y_score = compute_bidirectional_score(
        fwd_pred_df, bwd_pred_df, df, feature_list, n_total,
        score_method, agg_method, fusion_method,
    )
    y_score = smooth_score(y_score, SMOOTH_WINDOW)

    result = get_metrics(y_score, label,
                         slidingWindow=VUS_SLIDING_WINDOW, version=VUS_VERSION, thre=VUS_THRE)

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

    print("\n" + "=" * 50)
    print(f"FINAL AVERAGE METRICS - BIDIRECTIONAL "
          f"(score={score_method.upper()}, fusion={fusion_method.upper()})")
    print("=" * 50)
    print(f"Mean AUROC:   {np.mean(results['AUROC']):.4f}")
    print(f"Mean AUPRC:   {np.mean(results['AUPRC']):.4f}")
    print(f"Mean VUS-ROC: {np.mean(results['VUS-ROC']):.4f}")
    print(f"Mean VUS-PR:  {np.mean(results['VUS-PR']):.4f}")
    print("=" * 50)
    print(f"\n[SUCCESS] Per-file results saved to: {save_path}")


# -------------------- MAIN --------------------
def main():
    set_seed(SEED)
    args = parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print("Device:", device, "| CUDA Available:", torch.cuda.is_available())

    print(
        f"Config: direction=BIDIRECTIONAL model={args.model_size} score={args.score_method} "
        f"agg={args.agg_method} fusion={args.fusion_method} "
        f"context={args.context_length} horizon={args.horizon} "
        f"windows_per_batch={args.windows_per_batch}"
    )

    print("Loading Toto...")
    model = load_model(args.model_size, device)

    patch_size = model.config.patch_size
    if args.context_length % patch_size != 0:
        raise ValueError(
            f"--context_length {args.context_length} must be divisible by patch_size {patch_size}."
        )

    file_list = glob.glob(args.data_pattern)
    if not file_list:
        print(f"No files found matching pattern: {args.data_pattern}")
        return
    print(f"Found {len(file_list)} files to process.")

    results = defaultdict(list)

    for file_path in tqdm(file_list,
                          desc=f"Processing Files [BIDIRECTIONAL] ({args.score_method}/{args.fusion_method})",
                          position=0):
        file_name = os.path.basename(file_path).replace(".csv", "")
        metrics = process_file(
            model, file_path, args.score_method, args.agg_method, args.fusion_method,
            args.context_length, args.horizon, args.windows_per_batch, device,
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
# python toto_bidirectional.py --score_method interval --fusion_method max \
#     --data_pattern "./mTSBench/GHL/*test.csv" --save_path ghl_bidir_interval_max.csv
#
# Fusion ablations:
# python toto_bidirectional.py --fusion_method mean     --save_path ghl_bidir_mean.csv
# python toto_bidirectional.py --fusion_method min      --save_path ghl_bidir_min.csv
# python toto_bidirectional.py --fusion_method fwd_only --save_path ghl_bidir_fwd_only.csv
# python toto_bidirectional.py --fusion_method bwd_only --save_path ghl_bidir_bwd_only.csv