import pandas as pd
import numpy as np


def train_test_split(X,
                     y,
                     test_size=0.25,
                     train_size=None,
                     shuffle=True,
                     random_state=None,
                     stratify=None):
    
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise ValueError("y must be a pandas Series or DataFrame")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")

    n_samples = len(X)

    if train_size is None:
        train_size = 1 - test_size

    if not (0 < test_size and test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    if not (0 < train_size and train_size < 1):
        raise ValueError("train_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)

    if stratify is not None:
        unique_classes = stratify.unique()
        train_indices = []
        test_indices = []

        for class_label in unique_classes:
            class_label_idx = stratify[stratify == class_label].index.to_numpy()

            if shuffle:
                rng.shuffle(class_label_idx)

            class_label_test_size = int(len(class_label_idx) * test_size)

            test_indices.extend(class_label_idx[:class_label_test_size])
            train_indices.extend(class_label_idx[class_label_test_size:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

    else:
        indices = np.arange(n_samples)

        if shuffle:
            rng.shuffle(indices)

        test_count = int(n_samples * test_size)

        test_indices = indices[:test_count]
        train_indices = indices[test_count:]

    X_train = X.iloc[train_indices].reset_index(drop=True)
    X_test = X.iloc[test_indices].reset_index(drop=True)

    y_train = y.iloc[train_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
