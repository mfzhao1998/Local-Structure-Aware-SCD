"""Metric helpers used by semantic change detection evaluation."""

import numpy as np


def _fast_histogram(prediction, target, num_classes):
    valid = (prediction >= 0) & (prediction < num_classes)
    return np.bincount(
        num_classes * prediction[valid].astype(int) + target[valid],
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def get_hist(prediction, target, num_classes):
    histogram = np.zeros((num_classes, num_classes))
    histogram += _fast_histogram(
        prediction.flatten(), target.flatten(), num_classes
    )
    return histogram


def cal_kappa(histogram):
    if histogram.sum() == 0:
        return 0
    observed = np.diag(histogram).sum() / histogram.sum()
    expected = (
        np.matmul(histogram.sum(1), histogram.sum(0).T)
        / histogram.sum() ** 2
    )
    if expected == 1:
        return 0
    return (observed - expected) / (1 - expected)
