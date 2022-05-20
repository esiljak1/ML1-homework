import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    knn = KNearestNeighborsClassifier()
    #TODO: use the `GridSearchCV` meta-classifier and search over different values of `k`!
    # include the `return_train_score=True` option to get the training accuracies
    clf = ...
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")
    print(f"Dataset {idx}: {clf.best_params_}")

    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    # TODO you should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.show()
    plt.figure()
    plt.plot(clf.cv_results_['mean_train_score'])
    plt.plot(clf.cv_results_['mean_test_score'])
    plt.show()

