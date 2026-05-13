"""
FITS helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

FITS is a frequency-domain interpolation reconstructor. Per-timestep anomaly
scores are squared reconstruction errors averaged across features.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['FITS'].
The FITS class hardcodes EarlyStoppingTorch(patience=3) internally and does
not expose it as an argument, so we don't pass it.

Naming note: FITS uses `win_size` (not `window_size`) and `input_c` (not
`feats`), so the kwargs at the call site differ from CNN/LSTMAD/OmniAnomaly.
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import contextlib
import os
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.FITS import FITS
if not hasattr(FITS, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    FITS.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
FITS_HP = dict(
    win_size=100,            # fixed by optimal HP; not auto-detected
    DSR=4,                   # downsampling ratio; seq_len = win_size // DSR
    individual=True,
    input_c=1,               # overridden per task from data.shape[1]
    cut_freq=12,
    batch_size=128,
    epochs=50,
    lr=0.001,
    validation_size=0.2,
)


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout AND stderr during fit/decision_function.

    The model prints a GPU-detection banner to stdout and shows tqdm
    progress bars on stderr. The outer runner's tqdm bar is unaffected
    because it lives outside this context.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def anomaly_FITS(data_train: np.ndarray,
                 data_test: np.ndarray,
                 labels_test: np.ndarray,
                 sliding_window: int):
    """
    Fit FITS on data_train and evaluate on data_test.

    Note: sliding_window here is used ONLY for get_metrics range-based scoring.
    The model's own sequence length is FITS_HP['win_size'] = 100,
    fixed by the optimal HP dict and not auto-detected.

    Raises ValueError if the train or test region is too short to support
    win_size (rather than silently shrinking, which would deviate from the
    optimal HP).
    """
    input_c = data_train.shape[1]
    win_size = FITS_HP["win_size"]

    # ReconstructDataset needs win_size rows in each split
    min_train = int(np.ceil(win_size / (1 - FITS_HP["validation_size"])))
    if data_train.shape[0] < min_train:
        raise ValueError(
            f"Train too short ({data_train.shape[0]} rows) for "
            f"win_size={win_size}, validation_size={FITS_HP['validation_size']}. "
            f"Need >= {min_train}."
        )
    if data_test.shape[0] < win_size:
        raise ValueError(
            f"Test too short ({data_test.shape[0]} rows) for win_size={win_size}."
        )

    with _suppress_output():
        detector = FITS(
            win_size=win_size,
            DSR=FITS_HP["DSR"],
            individual=FITS_HP["individual"],
            input_c=input_c,
            batch_size=FITS_HP["batch_size"],
            cut_freq=FITS_HP["cut_freq"],
            epochs=FITS_HP["epochs"],
            lr=FITS_HP["lr"],
            validation_size=FITS_HP["validation_size"],
        )
        detector.fit(data_train)
        anomaly_score = detector.decision_function(data_test)

    anomaly_score = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        anomaly_score.reshape(-1, 1)
    ).ravel()

    pred = anomaly_score > (np.mean(anomaly_score) + 3 * np.std(anomaly_score))

    evaluation_result = get_metrics(
        anomaly_score,
        labels_test,
        slidingWindow=sliding_window,
        pred=pred,
    )

    return anomaly_score, evaluation_result
