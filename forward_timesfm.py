import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import uniform_filter1d

import timesfm
# from VUS_ROC_VUS_PR.metrics import get_metrics
from TSB_AD.evaluation.metrics import get_metrics


import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.set_float32_matmul_precision("high")


# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="TimesFM SMD Anomaly Detection (VUS metrics)")
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.2,
        help="Train/Test split ratio (e.g., 0.2 means 20%% train, 80%% test)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="TimesFM will predict timestamps"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=512,
        help="Number of past timestamps to use as context for predictions"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--score_method",
        type=str,
        default="interval",
        choices=["mse", "interval", "normalized_deviation", "smape"],
        help=(
            "Anomaly scoring method per feature:\n"
            "  mse                 - squared error vs median\n"
            "  interval            - violation beyond [0.1, 0.9] quantile band\n"
            "  normalized_deviation- |actual - median| / band_width\n"
            "  smape               - symmetric MAPE vs median"
        )
    )
    parser.add_argument(
        "--agg_method",
        type=str,
        default="topk_mean",
        choices=["l2", "max", "mean", "topk_mean"],
        help=(
            "How to aggregate per-feature scores into a single time-series score:\n"
            "  l2        - L2 norm\n"
            "  max       - maximum across features\n"
            "  mean      - mean across features\n"
            "  topk_mean - mean of top-k features"
        )
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Uniform smoothing window for final anomaly score (1 = no smoothing)"
    )
    parser.add_argument(
        "--sliding_window_VUS",
        type=int,
        default=100,
        help="Sliding-window size used by VUS metrics"
    )
    parser.add_argument(
        "--vus_version",
        type=str,
        default="opt",
        choices=["opt", "opt_mem"],
        help="VUS computation backend"
    )
    parser.add_argument(
        "--vus_thre",
        type=int,
        default=250,
        help="Number of thresholds used in VUS curve generation"
    )
    return parser.parse_args()


# -------------------- GPU SETUP --------------------
args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# -------------------- LOAD TIMESFM --------------------
# max_context / max_horizon are baked in at compile time, so size them above
# the CLI args (defaults 1024 / 256 are fine for context=512, horizon=100).
MAX_CONTEXT = max(1024, args.context_length)
MAX_HORIZON = max(256,  args.horizon)

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(
    timesfm.ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=MAX_HORIZON,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# TimesFM quantile_forecast layout: (batch, horizon, 10) = [mean, q10, q20, ..., q90]
Q10_IDX, Q50_IDX, Q90_IDX = 1, 5, 9


# -------------------- DATA PREPARATION --------------------
def prepare_df_test(df):
    df = df.copy()
    df["timestamp"] = pd.date_range(
        start="2000-02-01",
        periods=len(df),
        freq="1s"
    )
    df = df.sort_values("timestamp")

    ts = df["timestamp"]
    assert ts.is_monotonic_increasing
    assert ts.diff().dropna().nunique() == 1

    return df


def split_dataset(df, split_ratio):
    split_idx = int(len(df) * split_ratio)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    return df_train, df_test


# -------------------- PREDICTION --------------------
def _forecast_one_window(df_context, feature_list, horizon, context_length, window_id):
    """Run one TimesFM forecast over all features. Returns a DataFrame with the
    same columns Chronos's predict_df produces downstream:
    id, timestamp, target_name, 0.1, 0.5, 0.9, window_id."""

    # Batched univariate inputs, sliced to context_length.
    inputs = [
        df_context[f].to_numpy(dtype=np.float32)[-context_length:]
        for f in feature_list
    ]

    _, quantile_forecast = model.forecast(horizon=horizon, inputs=inputs)
    # quantile_forecast shape: (num_features, horizon, 10)

    # Future timestamps matching prepare_df_test's 1s frequency.
    last_ts   = df_context["timestamp"].iloc[-1]
    future_ts = pd.date_range(
        start=last_ts + pd.Timedelta(seconds=1),
        periods=horizon,
        freq="1s",
    )

    frames = []
    for i, feat in enumerate(feature_list):
        frames.append(pd.DataFrame({
            "id":          "SMD",
            "timestamp":   future_ts,
            "target_name": feat,
            "0.1":         quantile_forecast[i, :, Q10_IDX],
            "0.5":         quantile_forecast[i, :, Q50_IDX],
            "0.9":         quantile_forecast[i, :, Q90_IDX],
            "window_id":   window_id,
        }))
    return pd.concat(frames, ignore_index=True)


def generate_prediction(df_train, df_test, feature_list, prediction_length, context_length):
    window_length    = prediction_length
    all_predictions  = []
    num_windows      = len(df_test) // window_length
    remainder        = len(df_test) %  window_length
    total_windows    = num_windows + (1 if remainder > 0 else 0)

    pbar = tqdm(
        total=total_windows,
        desc="  forecasting windows",
        unit="win",
        leave=False,
        position=1,
    )

    for i in range(num_windows):
        start = i * window_length

        if i == 0:
            df_train_window = df_train.copy()
        else:
            df_past_test    = df_test.iloc[:start].copy()
            df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        all_predictions.append(
            _forecast_one_window(
                df_train_window, feature_list, window_length, context_length, i
            )
        )
        pbar.update(1)

    if remainder > 0:
        start           = num_windows * window_length
        df_past_test    = df_test.iloc[:start].copy()
        df_train_window = pd.concat([df_train, df_past_test], ignore_index=True)

        all_predictions.append(
            _forecast_one_window(
                df_train_window, feature_list, remainder, context_length, num_windows
            )
        )
        pbar.update(1)

    pbar.close()
    return pd.concat(all_predictions, ignore_index=True)


# -------------------- ANOMALY SCORING --------------------
def compute_feature_score(y_actual, group_df, method="mse"):
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


# -------------------- PATHS --------------------
# data_path = "/home/rajib/mTSBench/Datasets/mTSBench/SMD/*test.csv"
# data_path = "./mTSBench/SMD/SMD_machine-1-2_test.csv"
data_path = "./mTSBench/SMD/*test.csv"



# -------------------- MAIN LOOP --------------------
file_list         = glob.glob(data_path)
dic_for_each_file = defaultdict(list)
prediction_length = args.horizon
context_length    = args.context_length


for f in tqdm(file_list, desc="Processing SMD files", unit="file"):
    file_name = os.path.basename(f).replace(".csv", "")
    print(f"\nProcessing: {file_name}")

    df_original = pd.read_csv(f)
    df_original = prepare_df_test(df_original)

    feature_list = [
        c for c in df_original.columns
        if c not in ["timestamp", "is_anomaly"]
    ]

    df_train, df_test = split_dataset(df_original, args.split_ratio)
    df_train["id"] = "SMD"
    df_test["id"]  = "SMD"

    prediction_df = generate_prediction(
        df_train, df_test, feature_list, prediction_length, context_length
    )

    anomaly_scores = {}
    for feature_name, group_df in prediction_df.groupby("target_name"):
        group_df = group_df.reset_index(drop=True)
        y_actual = df_test[feature_name].values
        anomaly_scores[feature_name] = compute_feature_score(
            y_actual, group_df, method=args.score_method
        )

    anomaly_df = pd.DataFrame(anomaly_scores)

    if args.score_method != "smape":
        anomaly_df = anomaly_df.apply(
            lambda col: pd.Series(robust_normalize(col.values)), axis=0
        )
    anomaly_df = anomaly_df.fillna(0)

    y_score = aggregate_scores(anomaly_df, method=args.agg_method)

    if args.smooth_window > 1:
        y_score = uniform_filter1d(y_score, size=args.smooth_window)

    y_true = df_test["is_anomaly"].values.astype(int)

    if y_true.sum() == 0:
        print(f"Skipping {file_name}: no anomalies in ground truth")
        continue

    evaluation_result = get_metrics(
        y_score, y_true,
        slidingWindow=args.sliding_window_VUS,
        version=args.vus_version,
        thre=args.vus_thre,
    )

    vus_roc = evaluation_result["VUS-ROC"]
    vus_pr  = evaluation_result["VUS-PR"]
    auroc   = evaluation_result["AUC-ROC"]
    auprc   = evaluation_result["AUC-PR"]
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | "
          f"VUS-ROC: {vus_roc:.4f} | VUS-PR: {vus_pr:.4f}")

    dic_for_each_file["file_name"].append(file_name)
    dic_for_each_file["AUROC"].append(auroc)
    dic_for_each_file["AUPRC"].append(auprc)
    dic_for_each_file["VUS-ROC"].append(vus_roc)
    dic_for_each_file["VUS-PR"].append(vus_pr)


# -------------------- SUMMARY --------------------
auroc_list   = dic_for_each_file["AUROC"]
auprc_list   = dic_for_each_file["AUPRC"]
vus_roc_list = dic_for_each_file["VUS-ROC"]
vus_pr_list  = dic_for_each_file["VUS-PR"]

mean_auroc   = float(np.mean(auroc_list))   if auroc_list   else float("nan")
mean_auprc   = float(np.mean(auprc_list))   if auprc_list   else float("nan")
mean_vus_roc = float(np.mean(vus_roc_list)) if vus_roc_list else float("nan")
mean_vus_pr  = float(np.mean(vus_pr_list))  if vus_pr_list  else float("nan")
print("\nFinished processing all SMD files")
print(f"Mean AUROC: {mean_auroc:.4f}")
print(f"Mean AUPRC: {mean_auprc:.4f}")
print(f"Mean VUS-ROC: {mean_vus_roc:.4f}")
print(f"Mean VUS-PR : {mean_vus_pr:.4f}")



# python forward.py --score_method interval --agg_method topk_mean
# python forward_timesfm.py    --score_method interval --agg_method topk_mean