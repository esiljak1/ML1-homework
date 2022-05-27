import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
    def __init__(self, k=1):
      self.y = None
      self.X = None
      self.k = k

    def fit(self, X, y):
        # TODO IMPLEMENT ME
        # store X and y
        self.X = X
        self.y = y
        return

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict(self, X):
        # TODO: assign class labels
        # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
        # pay close attention to the `axis` parameter of these methods
        # broadcasting is really useful for this task!
        # See https://numpy.org/doc/stable/user/basics.broadcasting.html
        y_pred = []
        for i in range(X.shape[0]):
            distances = [np.linalg.norm(X[i] - self.X[j]) for j in range(self.X.shape[0])]
            sorted_indices = np.argsort(distances)[:self.k]
            neighbor_labels = self.y[sorted_indices]
            labels, count = np.unique(neighbor_labels, return_counts=True)
            max_label_index = np.argmax(count)
            y_pred.append(labels[max_label_index])
        return np.array(y_pred)
