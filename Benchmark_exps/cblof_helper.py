"""
CBLOF helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

Note: TSB-AD's CBLOF model does NOT take a slidingWindow argument (unlike
PCA/IForest). It works on per-timestep feature vectors directly. The
sliding_window argument is only used by get_metrics for range-based scoring,
not by the detector itself.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['CBLOF'].
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.CBLOF import CBLOF
if not hasattr(CBLOF, "__sklearn_tags__"):
    from sklearn.base import BaseEstimator
    CBLOF.__sklearn_tags__ = BaseEstimator.__sklearn_tags__
    
from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
CBLOF_HP = dict(
    # Number of clusters for the underlying KMeans step.
    n_clusters=4,

    # Expected fraction of anomalies. Affects only the binary threshold,
    # not the raw anomaly scores. VUS-PR is unaffected by this value.
    contamination=0.1,

    # None -> use KMeans(n_clusters=n_clusters, random_state=random_state).
    clustering_estimator=None,

    # Coefficient for separating large vs small clusters by cumulative size.
    # Must be in (0, 1). Higher alpha = stricter "large cluster" definition.
    alpha=0.6,

    # Coefficient for separating large vs small clusters by size ratio.
    # Must be > 1.
    beta=5,

    # If True, multiply scores by cluster size. Default False per pyod.
    use_weights=False,

    check_estimator=False,
    random_state=0,
    normalize=True,
)


def anomaly_CBLOF(data_train: np.ndarray,
                  data_test: np.ndarray,
                  labels_test: np.ndarray,
                  sliding_window: int):
    """
    Fit CBLOF on data_train and evaluate on data_test.

    sliding_window is used only by get_metrics for range-based scoring;
    CBLOF itself takes no sliding-window argument.

    Returns
    -------
    anomaly_score : (n_test,) ndarray
    evaluation_result : dict
    """
    # CBLOF uses KMeans internally. Guard against asking for more clusters
    # than we have training samples.
    n_clusters = min(CBLOF_HP["n_clusters"], data_train.shape[0] - 1)

    detector = CBLOF(
        n_clusters=n_clusters,
        contamination=CBLOF_HP["contamination"],
        clustering_estimator=CBLOF_HP["clustering_estimator"],
        alpha=CBLOF_HP["alpha"],
        beta=CBLOF_HP["beta"],
        use_weights=CBLOF_HP["use_weights"],
        check_estimator=CBLOF_HP["check_estimator"],
        random_state=CBLOF_HP["random_state"],
        normalize=CBLOF_HP["normalize"],
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