"""
TimesFM 2.5 helper for F2A anomaly detection benchmark.

Protocol: ZERO-SHOT on test data. The training file is NOT used because
TimesFM is a pretrained foundation model -- there is nothing to fit per
task. We score the test region using rolling 1-step-ahead forecasts and
treat normalized residuals as the anomaly score.

WHY THIS HELPER LOOKS DIFFERENT FROM THE OTHERS

The previous version used `model.forecast(...)`, which under the hood
loops over inputs in Python and dispatches one CUDA call per input.
Looks like batching but isn't. On Kaggle T4 this produced near-zero
throughput.

This version uses `model.compiled_decode(horizon, inputs, masks)`, the
JIT-compiled kernel that takes a pre-batched (batch, context_len) numpy
array and returns the full batch's forecasts in one CUDA call. Explicit
batching with BATCH_SIZE=192 keeps the GPU saturated.

Speed comparison (OPPORTUNITY_S1-ADL2_test, 25k rows, 242 channels, T4):
    - model.forecast(): hours, no visible progress
    - compiled_decode batched: ~15 min

Loading: from_pretrained() is used here. Earlier it failed with a
`proxies` TypeError; the fix on Kaggle was reinstalling timesfm fresh
from main, which gave a version where from_pretrained works correctly.
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import os
import threading
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from TSB_AD.evaluation.metrics import get_metrics

# Avoid CUDA memory fragmentation across many batches.
# Set BEFORE importing torch (torch reads this at import time).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
TIMESFM_HP = dict(
    # Number of past timesteps used as context for each forecast.
    context_len=96,

    # Forecast horizon. 1 = one-step-ahead, standard for AD scoring.
    horizon=1,

    # Number of windows processed per GPU batch. Higher = better GPU
    # utilization, more VRAM. 192 was tuned for T4 with context_len=96
    # and per-feature univariate forecasting (each feature is its own
    # input sequence).
    batch_size=128,

    # Hugging Face checkpoint id.
    checkpoint_id="google/timesfm-2.5-200m-pytorch",
)


# -----------------------------------------------------------------------------
# Model singleton (loaded once, reused across all task calls)
# -----------------------------------------------------------------------------
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_WARMED = False


# def _get_model():
#     """Load TimesFM 2.5 once and cache it. Thread-safe."""
#     global _MODEL
#     if _MODEL is not None:
#         return _MODEL

#     with _MODEL_LOCK:
#         if _MODEL is not None:
#             return _MODEL

#         import torch
#         import timesfm
#         from timesfm import TimesFM_2p5_200M_torch

#         torch.set_float32_matmul_precision("high")

#         print("[TimesFM] Loading TimesFM 2.5 checkpoint... (one-time)")
#         model = TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_HP["checkpoint_id"])

#         print(f"[TimesFM] Compiling for context_len={TIMESFM_HP['context_len']}, "
#               f"horizon={TIMESFM_HP['horizon']}...")
#         forecast_config = timesfm.configs.ForecastConfig(
#             max_context=TIMESFM_HP["context_len"],
#             max_horizon=TIMESFM_HP["horizon"],
#         )
#         model.compile(forecast_config=forecast_config)

#         print("[TimesFM] Loaded.")
#         _MODEL = model
#         return _MODEL



def _get_model():
    """Load TimesFM 2.5 once and cache it. Thread-safe."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        import torch
        import timesfm
        from timesfm import TimesFM_2p5_200M_torch

        torch.set_float32_matmul_precision("high")

        # =====================================================================
        # MONKEY PATCH START: Fix the 'proxies' TypeError caused by newer HF Hub
        # =====================================================================
        original_init = TimesFM_2p5_200M_torch.__init__
        
        def patched_init(self, *args, **kwargs):
            # Only pass the arguments that TimesFM actually expects
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['torch_compile', 'config']}
            original_init(self, *args, **valid_kwargs)
            
        TimesFM_2p5_200M_torch.__init__ = patched_init
        # =====================================================================
        # MONKEY PATCH END
        # =====================================================================

        print("[TimesFM] Loading TimesFM 2.5 checkpoint... (one-time)")
        model = TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_HP["checkpoint_id"])

        print(f"[TimesFM] Compiling for context_len={TIMESFM_HP['context_len']}, "
              f"horizon={TIMESFM_HP['horizon']}...")
        forecast_config = timesfm.configs.ForecastConfig(
            max_context=TIMESFM_HP["context_len"],
            max_horizon=TIMESFM_HP["horizon"],
        )
        model.compile(forecast_config=forecast_config)

        print("[TimesFM] Loaded.")
        _MODEL = model
        return _MODEL



def _warmup_jit(model, n_features_for_warmup: int):
    """Run one dummy batch to trigger JIT compilation. Subsequent batches
    are fast. We pass n_features as the multiplier because the helper
    flattens (batch, n_features, context) into (batch * n_features, context)
    before calling compiled_decode -- that flattened width is what the JIT
    sees and shape-specializes on.
    """
    global _MODEL_WARMED
    if _MODEL_WARMED:
        return

    print("[TimesFM] Warming up JIT compiler...")
    dummy_inputs = np.zeros(
        (TIMESFM_HP["batch_size"] * n_features_for_warmup, TIMESFM_HP["context_len"]),
        dtype=np.float32,
    )
    dummy_masks = np.ones_like(dummy_inputs, dtype=bool)
    _ = model.compiled_decode(TIMESFM_HP["horizon"], dummy_inputs, dummy_masks)
    print("[TimesFM] Warmup complete.")
    _MODEL_WARMED = True


# -----------------------------------------------------------------------------
# Anomaly scoring helper
# -----------------------------------------------------------------------------
def _residual_score(actuals: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    """Per-row anomaly score from residuals.

    Standardizes per-feature residuals (each feature has its own scale) and
    averages across features. This is the same scheme the Kaggle reference
    notebook used; it's safer than MSE for datasets with very differently
    scaled multivariate features (some channels can have huge dynamic
    range and dominate MSE-based scores).
    """
    residuals = np.abs(actuals - forecasts)            # (n_test, n_features)
    standardized = StandardScaler().fit_transform(residuals)
    return np.mean(standardized, axis=1)               # (n_test,)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def anomaly_TimesFM(data_train: np.ndarray,
                    data_test: np.ndarray,
                    labels_test: np.ndarray,
                    sliding_window: int):
    """
    Zero-shot anomaly scoring with TimesFM 2.5.

    NOTE: data_train is intentionally ignored. TimesFM is a pretrained
    foundation model and runs zero-shot on data_test alone.

    Algorithm (per the working Kaggle notebook):
      1. Build sliding-window contexts of length context_len from data_test
         using stride tricks (no copy).
      2. For each batch of `batch_size` windows, flatten across features
         and call compiled_decode for the entire batch in one CUDA call.
      3. Reshape forecasts back to (n_windows, n_features), compare with
         actuals, score by standardized absolute residuals averaged
         across features.
    """
    del data_train

    import torch

    model = _get_model()

    n_test, n_features = data_test.shape
    context_len = TIMESFM_HP["context_len"]
    horizon = TIMESFM_HP["horizon"]
    batch_size = TIMESFM_HP["batch_size"]

    # Need at least context_len + horizon rows in the test data
    valid_samples = n_test - context_len
    if valid_samples <= 0:
        raise ValueError(
            f"Test region too short ({n_test} rows) for "
            f"context_len={context_len}."
        )

    # Warm up the JIT once per python process. Specialize on the same
    # n_features the first real batch will use; subsequent calls with a
    # different n_features will trigger a brief recompile (still cheaper
    # than no warmup).
    _warmup_jit(model, n_features)

    # Build sliding windows as a view (no allocation) over data_test
    # Shape after sliding_window_view: (valid_samples, n_features, context_len)
    # We reshape per-batch below.
    windows = np.lib.stride_tricks.sliding_window_view(
        data_test, window_shape=(context_len,), axis=0
    )[:valid_samples]
    # windows.shape == (valid_samples, n_features, context_len)

    actuals = data_test[context_len:]                          # (valid_samples, n_features)
    predictions = np.empty((valid_samples, n_features), dtype=np.float32)

    # Process in fixed-size batches
    # ADDED TQDM HERE to see batch-level progress
    for start in tqdm(range(0, valid_samples, batch_size), desc="TimesFM Batches", leave=False):
        end = min(start + batch_size, valid_samples)
        cur = end - start


        # Flatten (cur, n_features, context_len) -> (cur * n_features, context_len)
        # Each feature is treated as its own univariate context for TimesFM.
        batch = windows[start:end].reshape(-1, context_len).astype(np.float32, copy=False)
        masks = np.ones_like(batch, dtype=bool)

        forecast_output = model.compiled_decode(horizon, batch, masks)
        # forecast_output[0] is the point forecast: shape (cur * n_features, horizon)
        flat_pf = forecast_output[0][:, 0]                     # take horizon=0
        predictions[start:end] = flat_pf.reshape(cur, n_features)

        # Defragment between batches; on T4 this prevents OOM on long files.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-row anomaly scores from standardized residuals
    raw_scores = _residual_score(actuals, predictions)         # (valid_samples,)

    # Pad back to original test length: first context_len timesteps have
    # no forecast; fill with the first valid score.
    full_scores = np.empty(n_test, dtype=np.float32)
    full_scores[:context_len] = raw_scores[0]
    full_scores[context_len:] = raw_scores

    # Min-max normalize to [0, 1] for metric consistency with other helpers
    full_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        full_scores.reshape(-1, 1)
    ).ravel()

    pred = full_scores > (np.mean(full_scores) + 3 * np.std(full_scores))

    evaluation_result = get_metrics(
        full_scores,
        labels_test,
        slidingWindow=sliding_window,
        pred=pred,
    )

    return full_scores, evaluation_result