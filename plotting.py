import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(X, clf, plot_step=0.05):
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                       np.arange(y_min, y_max, plot_step))

  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.5)

def plot_dataset(X_train, X_test, y_train, y_test):
  plt.scatter(x=X_train[:, 0], y=X_train[:, 1], c=y_train, label='train', cmap=plt.cm.RdYlBu)
  plt.scatter(x=X_test[:, 0], y=X_test[:, 1], c=y_test, label='test', cmap=plt.cm.RdYlBu, marker='x')
  plt.legend()

