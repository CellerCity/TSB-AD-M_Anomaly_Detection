"""
CNN helper for F2A anomaly detection benchmark.

Protocol: fit on data_train, score on data_test, evaluate against labels_test.

CNN is a 1D convolutional forecaster. Per-timestep anomaly scores are
squared forecast errors averaged across features.

Hyperparameters: from TSB_AD/HP_list.py -> Optimal_Multi_algo_HP_dict['CNN'].
The CNN class hardcodes EarlyStoppingTorch(patience=3) internally and does
not expose it as an argument, so we don't pass it.
"""

import sys
# sys.path.append('/home/rajib/TSB-AD')   # adjust if your TSB-AD path differs

import contextlib
import os
import warnings
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.models.CNN import CNN
from TSB_AD.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
CNN_HP = dict(
    window_size=100,            # fixed by optimal HP; not auto-detected
    pred_len=1,
    feats=1,                    # overridden per task from data.shape[1]
    num_channel=[32, 32, 40],
    batch_size=128,
    epochs=50,
    lr=0.0008,
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


def anomaly_CNN(data_train: np.ndarray,
                data_test: np.ndarray,
                labels_test: np.ndarray,
                sliding_window: int):
    """
    Fit CNN on data_train and evaluate on data_test.

    Note: sliding_window here is used ONLY for get_metrics range-based scoring.
    The model's own sequence length is CNN_HP['window_size'] = 100,
    fixed by the optimal HP dict and not auto-detected.

    Raises ValueError if the train or test region is too short to support
    window_size + pred_len (rather than silently shrinking, which would
    deviate from the optimal HP).
    """
    feats = data_train.shape[1]
    window_size = CNN_HP["window_size"]
    pred_len = CNN_HP["pred_len"]

    # ForecastDataset needs window_size + pred_len rows in each split
    min_train = int(np.ceil((window_size + pred_len) / (1 - CNN_HP["validation_size"])))
    if data_train.shape[0] < min_train:
        raise ValueError(
            f"Train too short ({data_train.shape[0]} rows) for "
            f"window_size={window_size}, pred_len={pred_len}, "
            f"validation_size={CNN_HP['validation_size']}. Need >= {min_train}."
        )
    if data_test.shape[0] < window_size + pred_len:
        raise ValueError(
            f"Test too short ({data_test.shape[0]} rows) for "
            f"window_size={window_size}, pred_len={pred_len}."
        )

    with _suppress_output():
        detector = CNN(
            window_size=window_size,
            pred_len=pred_len,
            batch_size=CNN_HP["batch_size"],
            epochs=CNN_HP["epochs"],
            lr=CNN_HP["lr"],
            feats=feats,
            num_channel=CNN_HP["num_channel"],
            validation_size=CNN_HP["validation_size"],
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
