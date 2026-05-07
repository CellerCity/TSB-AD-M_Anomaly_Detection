"""
PCA helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.
This is uniform across all six helpers per project methodology.

PCA fits a low-rank "normal" subspace from training data and scores test
points by their distance from that subspace (using minor components).

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['PCA'].
The optimal config specifies n_components=100. We cap that per-task at the
rank of the windowed feature matrix to handle short / low-dimensional series
that can't support 100 components.
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.PCA import PCA
if not hasattr(PCA, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    PCA.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
PCA_HP = dict(
    # Auto-detected per task; placeholder here.
    slidingWindow=100,

    sub=True,                # work on sub-sequences (windowed view)
    n_components=0.25,        # optimal HP
    n_selected_components=None,
    contamination=0.1,
    copy=True,
    whiten=False,
    svd_solver="auto",
    tol=0.0,
    iterated_power="auto",
    random_state=0,
    weighted=True,
    standardization=True,
    zero_pruning=True,
    normalize=True,
)



def anomaly_PCA(data_train: np.ndarray,
                data_test: np.ndarray,
                labels_test: np.ndarray,
                sliding_window: int):
    """
    Fit PCA on data_train and evaluate on data_test.

    Parameters
    ----------
    data_train : (n_train, d) ndarray
        Training data (without labels) used to learn the normal subspace.
    data_test : (n_test, d) ndarray
        Test data to score.
    labels_test : (n_test,) ndarray
        Binary labels (0/1) for the test data.
    sliding_window : int
        Auto-detected window length for the detector AND for get_metrics
        range-based scoring.

    Returns
    -------
    anomaly_score : (n_test,) ndarray
        Min-max normalized scores in [0, 1].
    evaluation_result : dict
        Metrics dict from TSB-AD's get_metrics.
    """
    # Cap n_components against what the train matrix can support after windowing
    n_components = PCA_HP["n_components"]
    n_selected = PCA_HP["n_selected_components"]


    detector = PCA(
        slidingWindow=sliding_window,
        sub=PCA_HP["sub"],
        n_components=PCA_HP["n_components"],   # pass through unchanged
        n_selected_components=PCA_HP["n_selected_components"],
        contamination=PCA_HP["contamination"],
        copy=PCA_HP["copy"],
        whiten=PCA_HP["whiten"],
        svd_solver=PCA_HP["svd_solver"],
        tol=PCA_HP["tol"],
        iterated_power=PCA_HP["iterated_power"],
        random_state=PCA_HP["random_state"],
        weighted=PCA_HP["weighted"],
        standardization=PCA_HP["standardization"],
        zero_pruning=PCA_HP["zero_pruning"],
        normalize=PCA_HP["normalize"],
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