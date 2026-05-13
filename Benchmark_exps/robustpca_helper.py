"""
RobustPCA helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

How it differs from vanilla PCA:
    Vanilla PCA finds a low-rank subspace by minimizing reconstruction error.
    Outliers can pull the principal components toward themselves, polluting
    the "normal" subspace.
    RobustPCA decomposes the data matrix D = L + S, where L is low-rank
    (the clean structure) and S is sparse (the anomalies). It then fits a
    standard PCA on L and uses |D - L| as the anomaly score per timestep.

Notes on this TSB-AD wrapper:
- The class does NOT take a slidingWindow argument. It operates on raw
  per-timestep feature vectors. The sliding_window argument is only used
  by get_metrics for range-based scoring.
- It does NOT expose contamination, random_state, or verbose. The PCP
  optimization in Robust_PCA is deterministic given the same input.
- The inner Robust_PCA.fit() has a hardcoded print() every 100 iterations.
  We silence it via a stdout redirect.
- ZERO-PRUNING GOTCHA: when zero_pruning=True, the wrapper drops zero
  columns of the TRAINING data during fit, then fits the inner PCA on the
  pruned matrix. decision_function does NOT internally re-prune, so we
  must apply the SAME pruning (using the train file's zero-column mask)
  to data_test before calling decision_function.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['RobustPCA'].
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import contextlib
import os
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.RobustPCA import RobustPCA
if not hasattr(RobustPCA, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    RobustPCA.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
ROBUSTPCA_HP = dict(
    # Maximum iterations of the PCP optimization.
    max_iter=1000,

    # Components for the inner PCA fitted on the low-rank matrix L.
    # None = use as many components as L has columns (after zero-pruning).
    n_components=None,

    # Drop columns that are entirely zero before fitting.
    zero_pruning=True,
)


@contextlib.contextmanager
def _suppress_stdout():
    """Silence Robust_PCA.fit()'s hardcoded per-100-iter prints."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def anomaly_RobustPCA(data_train: np.ndarray,
                      data_test: np.ndarray,
                      labels_test: np.ndarray,
                      sliding_window: int):
    """
    Fit RobustPCA on data_train and evaluate on data_test.

    sliding_window is used only by get_metrics; RobustPCA itself takes
    no sliding-window argument.

    Returns
    -------
    anomaly_score : (n_test,) ndarray
    evaluation_result : dict
    """
    detector = RobustPCA(
        max_iter=ROBUSTPCA_HP["max_iter"],
        n_components=ROBUSTPCA_HP["n_components"],
        zero_pruning=ROBUSTPCA_HP["zero_pruning"],
    )

    # Fit on the train region. PCP runs an SVD per iteration on the train
    # matrix, so this can be slow on long files.
    with _suppress_stdout():
        detector.fit(data_train)

    # Apply the SAME zero-column mask to the test data that the wrapper
    # applied to the training data, so dimensions match the fitted PCA.
    if detector.zero_pruning:
        non_zero_columns = np.any(data_train != 0, axis=0)
        data_test_for_score = data_test[:, non_zero_columns]
    else:
        data_test_for_score = data_test

    anomaly_score = detector.decision_function(data_test_for_score)

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