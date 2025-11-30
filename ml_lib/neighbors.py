import numpy as np
import pandas as pd

from .metrics import euclidean_distance, manhattan_distance, accuracy_score


class KNeighborsClassifier:

    def __init__(self, n_neighbors=5, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric.lower()
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.reset_index(drop=True)
        self.y_train = pd.Series(y).reset_index(drop=True)
        return self

    def _compute_distance(self, a, b):
        if self.metric == "euclidean":
            return euclidean_distance(a, b)
        elif self.metric == "manhattan":
            return manhattan_distance(a, b)
        else:
            raise ValueError("Unknown metric. Only 'euclidean' and 'manhattan' are available.")

    def predict(self, X):        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Impossible to predict on empty X_train and y_train.")

        predictions = []
        X_values = X.to_numpy()
        X_train_values = self.X_train.to_numpy()

        for x in X_values:
            distances = []

            for train_point in X_train_values:
                d = self._compute_distance(x, train_point)
                distances.append(d)

            neighbor_indices = np.argsort(distances)[:self.n_neighbors]

            neighbor_labels = self.y_train.iloc[neighbor_indices]
            most_common_label = neighbor_labels.value_counts().idxmax()

            predictions.append(most_common_label)

        return pd.Series(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
