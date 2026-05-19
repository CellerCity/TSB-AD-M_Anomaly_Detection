# TimesFM 2.5 zero-shot forecasts for TFRBench.
#
# Reads every *_public.json file in --data_dir and writes a submission
# directory matching the TFRBench schema:
#
#   my_submission/
#   ├── metadata.json
#   ├── NYC_Taxi.json
#   ├── amazon.json
#   └── ...
#
# Each output JSON is a list of:
#   {"id": ..., "Reasoning": "Zero shot inference only", "Prediction": [[...], ...]}
#
# TimesFM is univariate, so for multi-channel samples we forecast each channel
# independently and stack along the channel axis.

import os
import json
import glob
import argparse
import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

import timesfm
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------- CONFIG --------------------
SEED        = 2024
MAX_CONTEXT = 1024   # TimesFM 2.5 hard caps (set at compile time).
MAX_HORIZON = 256

DEFAULT_REASONING = "Zero shot inference only"
METADATA = {
    "model_name":  "TimesFM-2.5-200M (zero-shot)",
    "link":        "https://github.com/google-research/timesfm",
    "description": "Zero-shot point forecasts from TimesFM 2.5 200M. No fine-tuning, no reasoning.",
}


# -------------------- REPRODUCIBILITY --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


# -------------------- ARGS --------------------
def parse_args():
    p = argparse.ArgumentParser(description="TimesFM 2.5 zero-shot submissions for TFRBench")
    p.add_argument("--data_dir", type=str, default="./my_local_data",
                   help="Directory containing TFRBench *_public.json files")
    p.add_argument("--output_dir", type=str, default="./tfrbench_submission",
                   help="Directory to write submission JSON files into")
    p.add_argument("--weights_path", type=str, default="./timesfm-weights",
                   help="Local TimesFM 2.5 checkpoint (or HF id)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Number of univariate series per model.forecast call")
    p.add_argument("--reasoning", type=str, default=DEFAULT_REASONING,
                   help="String written into every sample's Reasoning field")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process datasets even if the output JSON already exists")
    return p.parse_args()


# -------------------- MODEL --------------------
def load_model(weights_path):
    """Load TimesFM 2.5 and compile with our inference config (mirrors forward script)."""
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


# -------------------- FORECASTING CORE --------------------
def forecast_univariate(model, series_list, horizon, batch_size):
    """Forecast a list of 1-D float arrays to a common horizon.

    Returns a list of (horizon,) float32 arrays aligned with series_list.

    Handles horizon > MAX_HORIZON via autoregressive rollout: predict MAX_HORIZON,
    append predictions to context (clipped to MAX_CONTEXT), repeat. All series
    in the batch advance in lockstep so we keep batched calls throughout.
    """
    n = len(series_list)
    out = [np.zeros(horizon, dtype=np.float32) for _ in range(n)]

    # Pre-truncate contexts to MAX_CONTEXT (TimesFM would do this internally, but
    # doing it here keeps the rollout buffers small).
    contexts = [s[-MAX_CONTEXT:].astype(np.float32, copy=True) for s in series_list]

    filled = 0
    while filled < horizon:
        step = min(MAX_HORIZON, horizon - filled)

        for i in range(0, n, batch_size):
            chunk = contexts[i : i + batch_size]
            point_fc, _ = model.forecast(horizon=step, inputs=chunk)
            point_fc = np.asarray(point_fc)[:, :step]  # (b, step)

            for j, pred in enumerate(point_fc):
                out[i + j][filled : filled + step] = pred
                # Only roll context forward if we still have more steps to predict.
                if filled + step < horizon:
                    contexts[i + j] = np.concatenate(
                        [contexts[i + j], pred]
                    )[-MAX_CONTEXT:]

        filled += step

    return out


# -------------------- PER-FILE PIPELINE --------------------
def process_file(model, input_path, output_path, batch_size, reasoning):
    """Run TimesFM on every sample in one TFRBench *_public.json file and
    write a submission JSON for that dataset."""
    with open(input_path, "r") as f:
        samples = json.load(f)

    # Flatten samples into per-channel jobs, grouped by horizon length
    # (model.forecast takes a single horizon per call).
    horizon_groups = defaultdict(list)   # horizon -> list of (sample_idx, channel_idx, context_1d)
    sample_meta    = []                  # per sample: {"id", "n_channels", "horizon"}

    for s_idx, sample in enumerate(samples):
        hist = sample["historical_window"]
        data = np.asarray(hist["data"], dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)        # safety: treat scalar series as (T, 1)
        n_ch    = data.shape[1]
        horizon = len(sample["future_window_timestamps"])

        sample_meta.append({"id": sample["id"], "n_channels": n_ch, "horizon": horizon})
        for c in range(n_ch):
            horizon_groups[horizon].append((s_idx, c, data[:, c]))

    # Allocate (horizon, n_channels) prediction matrix per sample.
    sample_pred = [
        np.zeros((m["horizon"], m["n_channels"]), dtype=np.float32)
        for m in sample_meta
    ]

    base = os.path.basename(input_path)
    for horizon, jobs in tqdm(
        horizon_groups.items(),
        desc=f"  {base}",
        unit="hgrp",
        leave=False,
        position=1,
    ):
        series = [job[2] for job in jobs]
        preds  = forecast_univariate(model, series, horizon, batch_size)
        for (s_idx, c, _), pred in zip(jobs, preds):
            sample_pred[s_idx][:, c] = pred

    # Assemble submission objects in the original input order.
    submission = [
        {
            "id":         meta["id"],
            "Reasoning":  reasoning,
            "Prediction": pred.tolist(),
        }
        for meta, pred in zip(sample_meta, sample_pred)
    ]

    with open(output_path, "w") as f:
        json.dump(submission, f)


# -------------------- MAIN --------------------
def main():
    set_seed(SEED)
    args = parse_args()

    print("CUDA Available:", torch.cuda.is_available())
    print(f"Reading TFRBench inputs from: {args.data_dir}")
    print(f"Writing submission to:        {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Always (re)write metadata.json.
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(METADATA, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")

    input_files = sorted(glob.glob(os.path.join(args.data_dir, "*_public.json")))
    if not input_files:
        print(f"[WARN] No *_public.json files found in {args.data_dir}")
        return
    print(f"Found {len(input_files)} TFRBench input files.")

    print("Loading TimesFM 2.5 ...")
    model = load_model(args.weights_path)

    for input_path in tqdm(input_files, desc="Datasets", position=0):
        # NYC_Taxi_public.json -> NYC_Taxi.json
        out_name    = os.path.basename(input_path).replace("_public.json", ".json")
        output_path = os.path.join(args.output_dir, out_name)

        if os.path.exists(output_path) and not args.overwrite:
            tqdm.write(f"  Skipping {out_name} (already exists; pass --overwrite to redo)")
            continue

        process_file(model, input_path, output_path, args.batch_size, args.reasoning)

    print(f"\n[SUCCESS] Submission written to: {args.output_dir}")
    print("Zip the directory and submit via the form linked in the TFRBench README.")


if __name__ == "__main__":
    main()


# Default run:
# python timesfm_tfrbench_zeroshot.py --data_dir ./TFRBench --output_dir ./tfrbench_submission
#
# With a larger forecast batch (more channels per call -> faster on big GPUs):
# python timesfm_tfrbench_zeroshot.py --batch_size 128
#
# Re-run a dataset after editing:
# python timesfm_tfrbench_zeroshot.py --overwrite