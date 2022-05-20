import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
  svm = LinearSVM()
  # TODO use grid search to find suitable parameters!
  clf = ...
  clf.fit(X_train, y_train)

  # TODO Use the parameters you have found to instantiate a LinearSVM.
  # the `fit` method returns a list of scores that you should plot in order
  # to monitor the convergence. When does the classifier converge?
  svm = LinearSVM(...)
  scores = svm.fit(X_train, y_train)
  plt.plot(scores)
  test_score = clf.score(X_test, y_test)
  print(f"Test Score: {test_score}")

  # TODO plot the decision boundary!
