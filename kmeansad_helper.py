"""
KMeansAD helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

How KMeansAD works:
    The time series is sliced into overlapping windows of length window_size,
    taken every `stride` steps. Each window becomes one multivariate point.
    KMeans clusters these window-points. The anomaly score per window is its
    Euclidean distance to its assigned cluster centroid. Per-window scores
    are then projected back to per-timestep scores via averaging.

The class signature uses `window_size` (not `slidingWindow`) and `k` (not
`n_clusters`). KMeansAD does NOT have a separate decision_function; we use
fit + predict (predict re-windows the test data with the trained centroids).

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['KMeansAD'].
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import contextlib
import os
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.KMeansAD import KMeansAD
if not hasattr(KMeansAD, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    KMeansAD.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
KMEANSAD_HP = dict(
    # Number of clusters representing "types" of normal window patterns.
    k=10,

    # Window length is set per-task from the auto-detected sliding_window.
    # The value here is just a placeholder.
    window_size=40,

    # Stride between consecutive windows. 1 = maximum overlap.
    stride=1,

    # Parallelism. KMeansAD's wrapper takes n_jobs but does not forward it
    # to its internal KMeans. Leave at 1 for determinism.
    n_jobs=1,

    # Per-window z-score normalization before clustering.
    normalize=True,
)


@contextlib.contextmanager
def _suppress_stdout():
    """Silence print() inside KMeansAD's _preprocess_data and
    _custom_reverse_windowing, which print progress info that's not
    configurable via any flag.

    tqdm bars (used by the outer runner) write to stderr, so they're
    unaffected.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def anomaly_KMeansAD(data_train: np.ndarray,
                     data_test: np.ndarray,
                     labels_test: np.ndarray,
                     sliding_window: int):
    """
    Fit KMeansAD on data_train and evaluate on data_test.

    sliding_window is used both as the model's window_size AND for
    get_metrics range-based scoring.

    Returns
    -------
    anomaly_score : (n_test,) ndarray
    evaluation_result : dict
    """
    # Number of windows in the TRAIN region after expansion -- KMeans needs
    # at least k of these to form clusters. Cap k to be safe.
    n_train_windows = max(
        1,
        (data_train.shape[0] - sliding_window) // KMEANSAD_HP["stride"] + 1,
    )
    k = min(KMEANSAD_HP["k"], n_train_windows)

    # Test region must also have at least one full window
    n_test_windows = max(
        0,
        (data_test.shape[0] - sliding_window) // KMEANSAD_HP["stride"] + 1,
    )
    if n_test_windows < 1:
        raise ValueError(
            f"Test region too short ({data_test.shape[0]} rows) for "
            f"window_size={sliding_window}, stride={KMEANSAD_HP['stride']}."
        )

    detector = KMeansAD(
        k=k,
        window_size=sliding_window,
        stride=KMEANSAD_HP["stride"],
        n_jobs=KMEANSAD_HP["n_jobs"],
        normalize=KMEANSAD_HP["normalize"],
    )

    # KMeansAD has no decision_function; use fit + predict separately so we
    # can train on data_train and score data_test.
    with _suppress_stdout():
        detector.fit(data_train)
        anomaly_score = detector.predict(data_test)

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