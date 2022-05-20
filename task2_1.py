import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import plotting
from datasets import get_toy_dataset


def loss(w, b, C, X, y):
  # TODO: implement the loss function (eq. 1)
  # useful methods: np.sum, np.clip
  return np.inf


def grad(w, b, C, X, y):
  # TODO: implement the gradients with respect to w and b.
  # useful methods: np.sum, np.where, numpy broadcasting
  grad_w = ...
  grad_b = ...
  return grad_w, grad_b


class LinearSVM(BaseEstimator):

  def __init__(self, C=1, eta=1e-3, max_iter=1000):
    self.C = C
    self.max_iter = max_iter
    self.eta = eta

  def fit(self, X, y):
    # TODO: initialize w and b. Does the initialization matter?
    # convert y: {0,1} -> -1, 1
    y = np.where(y == 0, -1, 1)
    self.w = np.random.normal(size=X.shape[1])
    self.b = 0.
    loss_list = []

    for j in range(self.max_iter):
      # TODO: compute the gradients, update the weights, compute the loss
      grad_w, grad_b = grad(self.w, self.b, self.C, X, y)
      self.w = ...
      self.b = ...
      loss_list.append(loss(self.w, self.b, self.C, X, y))

    return loss_list

  def predict(self, X):
    # TODO: assign class labels to unseen data
    y_pred = ...
    # converting y_pred from {-1, 1} to {0, 1}
    return np.where(y_pred == -1, 0, 1)

  def score(self, X, y):
    # TODO: IMPLEMENT ME
    y_pred = self.predict(X)
    return np.mean(y_pred == y)
