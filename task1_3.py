import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier
from sklearn.model_selection import cross_val_score



if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
  for k in [1, 30, 100]:
    # TODO fit your KNearestNeighborsClassifier with k in {1, 30, 100} and plot the decision boundaries
    clf = ...
    # TODO you can use the `cross_val_score` method to manually perform cross-validation
    # TODO report mean cross-validated scores!
    test_score = clf.score(X_test, y_test)
    print(f"Test Score for k={k}: {test_score}")
    # TODO plot the decision boundaries!

  # TODO find the best parameters for the noisy dataset!
  knn = KNearestNeighborsClassifier()
  clf = ...
  # TODO The `cv_results_` attribute of `GridSearchCV` contains useful aggregate information
  # such as the `mean_train_score` and `mean_test_score`. Plot these values as a function of `k` and report the best
  # parameters. Is the classifier very sensitive to the choice of k?
