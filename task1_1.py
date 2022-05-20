import numpy as np
from sklearn.base import BaseEstimator


class KNearestNeighborsClassifier(BaseEstimator):
  def __init__(self, k=1):
    self.k = k

  def fit(self, X, y):
    # TODO IMPLEMENT ME
    # store X and y
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
    return
