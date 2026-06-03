# BACKWARD (TOTO 2.0, native multivariate):-
#
# Symmetric counterpart to the Toto FORWARD script.
# For each window [s, s+h), use the FUTURE rows [s+h, s+h+context_length)
# REVERSED as context, ask Toto to predict h steps in fake-time, then flip the
# quantile predictions back so row i aligns with real-time index s+i.
#
# When the future runs out near the end of the series, fall back to a forward
# forecast from the preceding past so every row of df still gets a score.
#
# Toto difference vs TimesFM: each window is fed as ONE multivariate tensor
# (batch, n_var, time) with series_ids all 0, so all features attend to each
# other (Chronos-style group attention). The reversed context is reversed
# jointly across channels, so cross-variate attention sees a time-reversed
# multivariate window -- the faithful analogue of the TimesFM backward pass.
#
# Metrics use fixed VUS parameters (matching the Toto FORWARD script):
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

# Toto checkpoint. Sizes: 4m | 22m | 313m | 1B | 2.5B
MODEL_SIZE = "4m"

# Toto quantile head knots = [0.1, 0.2, ..., 0.9]; forecast() returns
# shape (Q=9, batch, n_var, horizon). Index 4 is the median (0.5).
Q10_IDX, Q50_IDX, Q90_IDX = 0, 4, 8

# Mask is all-ones (dropna'd data), so set False on Ampere+ GPUs for the
# flash-attn speedup. Leave True on CPU / older GPUs (e.g. K80).
HAS_MISSING_VALUES = True

# Fixed VUS parameters for all evaluations.
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
    parser = argparse.ArgumentParser(description="Toto 2.0 Zero-Shot Backward Anomaly Detection (TSB-AD)")
    parser.add_argument("--data_pattern", type=str, default="./mTSBench/GHL/*test.csv",
                        help="Glob pattern for input test CSVs")
    parser.add_argument("--save_path", type=str, default="ghl_toto_backward_results.csv",
                        help="Where to save per-file metrics CSV")
    parser.add_argument("--context_length", type=int, default=512,
                        help="FUTURE timesteps used as (reversed) context (must be divisible by patch_size)")
    parser.add_argument("--horizon", type=int, default=128,
                        help="Number of timesteps scored per window")
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
    """context_stack: list of (n_var, ctx) float arrays, one per window (uniform ctx).
    Returns quantiles ndarray of shape (9, batch, n_var, horizon).
    series_ids all 0 -> full cross-variate attention. mask_stack (optional) marks
    padded positions; when present, has_missing_values is forced True.
    """
    target     = torch.from_numpy(np.stack(context_stack, axis=0)).to(device)  # (B, n_var, ctx)
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
    )  # (9, B, n_var, horizon)
    return quantiles.detach().cpu().numpy()


# -------------------- PREDICTION --------------------
def generate_backward_prediction(model, df, feature_list, n_total,
                                 context_length, horizon, windows_per_batch, device):
    """Run batched Toto multivariate forecasts using REVERSED FUTURE rows as context.

    For each window starting at `start` (step = horizon, walking 0 -> n_total):
      - Real-time target window is [start, start + h).
      - Real-time future context is [start + h, start + h + context_length), capped at n_total.
      - If at least `h` future rows are available, reverse them (jointly across all
        channels) and ask Toto to predict `horizon` steps in fake-time. After
        flipping back, row i aligns with real-time index start + i.
      - If the future is too short (final region), FALL BACK to a forward forecast
        from the preceding past so every row of df is scored.

    Returns a long-format DataFrame: target_name, t_idx, 0.1, 0.5, 0.9
    Coverage is the full range [0, n_total) thanks to the fallback.
    """
    starts = list(range(0, n_total, horizon))
    n_dim  = len(feature_list)
    patch_size = model.config.patch_size
    feat_matrix = df[feature_list].to_numpy(dtype=np.float32)  # (T, n_dim)

    q10_chunks, q50_chunks, q90_chunks, t_chunks = [], [], [], []
    fallback_count = 0

    pbar = tqdm(total=len(starts), desc="  backward windows",
                unit="win", leave=False, position=1)

    for i in range(0, len(starts), windows_per_batch):
        batch_starts   = starts[i : i + windows_per_batch]
        context_stack  = []   # each (n_var, ctx)
        mask_stack     = []
        valid_horizons = []
        flip_flags     = []   # True if window is backward (must flip predictions)

        for start in batch_starts:
            h = min(horizon, n_total - start)
            valid_horizons.append(h)

            fut_start     = start + h
            fut_end       = min(fut_start + context_length, n_total)
            fut_available = fut_end - fut_start

            if fut_available >= h:
                # Backward path: reverse the future rows (jointly across channels).
                ctx = feat_matrix[fut_start:fut_end][::-1, :].T.copy()  # (n_dim, ctx), positive strides
                flip_flags.append(True)
            else:
                # Fallback: forward forecast from preceding past.
                past_start = max(0, start - context_length)
                ctx = feat_matrix[past_start:start].T  # (n_dim, ctx)
                if ctx.shape[1] == 0:
                    # Degenerate: no future AND no past. Skip this window.
                    valid_horizons[-1] = 0
                    flip_flags.append(False)
                    context_stack.append(np.zeros((n_dim, patch_size), dtype=np.float32))
                    mask_stack.append(np.ones((n_dim, patch_size), dtype=bool))
                    continue
                flip_flags.append(False)
                fallback_count += 1

            # Snap context to a patch-multiple length (crop boundary-side, or pad+mask).
            ctx2, mask2 = fit_context_to_patch(np.ascontiguousarray(ctx), patch_size)
            context_stack.append(ctx2)
            mask_stack.append(mask2)

        # Contexts may differ in length (windows cropped to different multiples near
        # the ends). Toto batches a single tensor, so uneven lengths run as singletons.
        ctx_lens = [c.shape[1] for c in context_stack]
        uniform  = len(set(ctx_lens)) == 1

        if uniform:
            quant = _toto_forecast_batch(model, context_stack, horizon, device, mask_stack)  # (9,B,n_var,h)
            per_window_quant = [quant[:, b] for b in range(quant.shape[1])]      # list of (9,n_var,h)
        else:
            per_window_quant = []
            for c, m in zip(context_stack, mask_stack):
                q = _toto_forecast_batch(model, [c], horizon, device, [m])        # (9,1,n_var,h)
                per_window_quant.append(q[:, 0])                                  # (9,n_var,h)

        for wq, h, start, flip in zip(per_window_quant, valid_horizons, batch_starts, flip_flags):
            pbar.update(1)
            if h == 0:
                continue  # degenerate skip

            q10 = wq[Q10_IDX, :, :h]
            q50 = wq[Q50_IDX, :, :h]
            q90 = wq[Q90_IDX, :, :h]
            if flip:
                # Reversed-context predictions come out in reverse real-time order.
                q10 = q10[:, ::-1].copy()
                q50 = q50[:, ::-1].copy()
                q90 = q90[:, ::-1].copy()
            q10_chunks.append(q10)
            q50_chunks.append(q50)
            q90_chunks.append(q90)
            t_chunks.append(np.arange(start, start + h))

    pbar.close()

    if fallback_count > 0:
        print(f"  fallback (forward-from-past) windows: {fallback_count}")

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


# -------------------- ANOMALY SCORING (shared with forward) --------------------
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


def compute_anomaly_score(prediction_df, df, feature_list, score_method, agg_method):
    """Per-feature score -> robust normalize (skipped for sMAPE) -> aggregate."""
    anomaly_scores = {}
    for feature_name, group_df in prediction_df.groupby("target_name"):
        group_df = group_df.sort_values("t_idx").reset_index(drop=True)
        t_start = int(group_df["t_idx"].iloc[0])
        t_end   = int(group_df["t_idx"].iloc[-1]) + 1
        y_actual = df[feature_name].iloc[t_start:t_end].to_numpy(dtype=np.float32)
        anomaly_scores[feature_name] = compute_feature_score(y_actual, group_df, method=score_method)

    anomaly_df = pd.DataFrame(anomaly_scores)

    if score_method != "smape":
        anomaly_df = anomaly_df.apply(lambda col: pd.Series(robust_normalize(col.values)), axis=0)
    anomaly_df = anomaly_df.fillna(0)

    return aggregate_scores(anomaly_df, method=agg_method)


# -------------------- POST-PROCESSING --------------------
def smooth_score(y_score, smooth_window):
    """Backward (with fallback) already covers [0, n_total), so no padding -- just smooth."""
    if smooth_window > 1:
        return uniform_filter1d(y_score, size=smooth_window)
    return y_score


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, file_path, score_method, agg_method,
                 context_length, horizon, windows_per_batch, device):
    df, feature_list, label, n_total, n_dim = load_and_prepare(file_path)

    if label.sum() == 0:
        return None
    if n_total <= context_length:
        print(f"  Skipping (n_total={n_total} <= context_length={context_length})")
        return None

    prediction_df = generate_backward_prediction(
        model, df, feature_list, n_total,
        context_length, horizon, windows_per_batch, device,
    )

    y_score = compute_anomaly_score(prediction_df, df, feature_list, score_method, agg_method)
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
def summarize_and_save(results, save_path, score_method):
    if not results["file_name"]:
        print("No files were successfully processed.")
        return

    pd.DataFrame(results).to_csv(save_path, index=False)

    print("\n" + "=" * 50)
    print(f"FINAL AVERAGE METRICS - BACKWARD (Method: {score_method.upper()})")
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
        f"Config: direction=BACKWARD model={args.model_size} score={args.score_method} "
        f"agg={args.agg_method} context={args.context_length} horizon={args.horizon} "
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

    for file_path in tqdm(file_list, desc=f"Processing Files [BACKWARD] ({args.score_method})", position=0):
        file_name = os.path.basename(file_path).replace(".csv", "")
        metrics = process_file(
            model, file_path, args.score_method, args.agg_method,
            args.context_length, args.horizon, args.windows_per_batch, device,
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
# python toto_backward.py --score_method interval --data_pattern "./mTSBench/GHL/*test.csv" --save_path ghl_backward_interval.csv
#
# Ablation examples:
# python toto_backward.py --context_length 256  --horizon 32  --save_path ghl_bwd_c256_h32.csv
# python toto_backward.py --context_length 1024 --horizon 100 --save_path ghl_bwd_c1024_h100.csv