"""
OmniAnomaly helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.
Same as the unsupervised helpers; OmniAnomaly was already running this way
internally, just under a different argument convention.

OmniAnomaly is a GRU + variational autoencoder. Per-timestep anomaly scores
are reconstruction errors. Needs a GPU for reasonable runtime.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['OmniAnomaly'].
patience is NOT in the optimal HP dict; we use the class default of 3.
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import contextlib
import os
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.OmniAnomaly import OmniAnomaly
from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
OMNIANOMALY_HP = dict(
    win_size=100,            # fixed by optimal HP; not auto-detected
    feats=1,                 # overridden per task from data.shape[1]
    batch_size=128,
    epochs=50,
    patience=3,              # class default; not in optimal HP dict
    lr=0.002,
    validation_size=0.2,
)


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout AND stderr during fit/decision_function.

    OmniAnomaly prints the GPU detection banner to stdout and shows tqdm
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


def anomaly_OmniAnomaly(data_train: np.ndarray,
                         data_test: np.ndarray,
                         labels_test: np.ndarray,
                         sliding_window: int):
    """
    Fit OmniAnomaly on data_train and evaluate on data_test.

    Note: sliding_window here is used ONLY for get_metrics range-based scoring.
    The model's own sequence length is OMNIANOMALY_HP['win_size'] = 100,
    fixed by the optimal HP dict and not auto-detected.

    Raises ValueError if the train or test region is too short to support
    win_size=100 (rather than silently shrinking the window, which would
    deviate from the optimal HP).
    """
    feats = data_train.shape[1]
    win_size = OMNIANOMALY_HP["win_size"]

    # Both train and test must be large enough for the model's window
    min_train = int(np.ceil(win_size / (1 - OMNIANOMALY_HP["validation_size"])))
    if data_train.shape[0] < min_train:
        raise ValueError(
            f"Train too short ({data_train.shape[0]} rows) for "
            f"win_size={win_size}, validation_size={OMNIANOMALY_HP['validation_size']}. "
            f"Need >= {min_train}."
        )
    if data_test.shape[0] < win_size:
        raise ValueError(
            f"Test too short ({data_test.shape[0]} rows) for win_size={win_size}."
        )

    with _suppress_output():
        detector = OmniAnomaly(
            win_size=win_size,
            feats=feats,
            batch_size=OMNIANOMALY_HP["batch_size"],
            epochs=OMNIANOMALY_HP["epochs"],
            patience=OMNIANOMALY_HP["patience"],
            lr=OMNIANOMALY_HP["lr"],
            validation_size=OMNIANOMALY_HP["validation_size"],
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