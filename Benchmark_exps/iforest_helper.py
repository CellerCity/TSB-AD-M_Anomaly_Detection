"""
IForest helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

This is the template for all the unsupervised helpers. To produce
cblof_helper / robustpca_helper / kmeansad_helper, copy this file, rename
the class import and the function name, and replace the HP block.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['IForest'].
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.IForest import IForest
if not hasattr(IForest, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    IForest.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
IFOREST_HP = dict(
    slidingWindow=100,       # auto-detected per task; placeholder here
    n_estimators=25,
    sub=True,
    max_samples="auto",
    contamination=0.1,
    max_features=0.8, 
    bootstrap=False,
    n_jobs=1,
    behaviour="old",
    random_state=0,
    verbose=0,
    normalize=True,
)


def anomaly_IForest(data_train: np.ndarray,
                    data_test: np.ndarray,
                    labels_test: np.ndarray,
                    sliding_window: int):
    """
    Fit IForest on data_train and evaluate on data_test.

    Returns
    -------
    anomaly_score : (n_test,) ndarray
        Min-max normalized scores in [0, 1].
    evaluation_result : dict
        Metrics dict from TSB-AD's get_metrics.
    """
    detector = IForest(
        slidingWindow=sliding_window,
        n_estimators=IFOREST_HP["n_estimators"],
        sub=IFOREST_HP["sub"],
        max_samples=IFOREST_HP["max_samples"],
        contamination=IFOREST_HP["contamination"],
        max_features=IFOREST_HP["max_features"],
        bootstrap=IFOREST_HP["bootstrap"],
        n_jobs=IFOREST_HP["n_jobs"],
        behaviour=IFOREST_HP["behaviour"],
        random_state=IFOREST_HP["random_state"],
        verbose=IFOREST_HP["verbose"],
        normalize=IFOREST_HP["normalize"],
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