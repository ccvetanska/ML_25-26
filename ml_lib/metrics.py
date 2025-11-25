import numpy as np


def accuracy_score(y_true, y_pred, normalize=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if normalize:
        return np.sum(y_pred == y_true) / len(y_true)
    else:
        return np.sum(y_pred == y_true)


def euclidean_distance(point_one, point_two):
    p1 = np.asarray(point_one, dtype=float)
    p2 = np.asarray(point_two, dtype=float)

    if (p1.shape != p2.shape):
        raise ValueError(
            'point_one and point_two must have the same shape (dimensionality)')

    return np.sqrt(np.sum((p1 - p2)**2))

def manhattan_distance(point_one, point_two):
    p1 = np.asarray(point_one, dtype=float)
    p2 = np.asarray(point_two, dtype=float)

    if (p1.shape != p2.shape):
        raise ValueError(
            'point_one and point_two must have the same shape (dimensionality)')

    return np.sum(np.abs(p1 - p2))
