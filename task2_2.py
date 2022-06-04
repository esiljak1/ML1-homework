import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV
import numpy as np

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    print(X_train.shape)
    svm = LinearSVM()
    # TODO use grid search to find suitable parameters!

    """
    for c in 10.**np.arange(-4, 4):
        svm_new = LinearSVM(C=c)
        svm_new.fit(X_train, y_train)
        svm_new.score(X_test, y_test)

        plt.figure()
        plt.suptitle(f"Plot for C={c}")
        plt.xlabel("x")
        plt.ylabel("y")
        plotting.plot_decision_boundary(X_train, svm_new)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.show()
    """

    param_grid = dict(
      C=10.**np.arange(-4, 4),
      eta=10.**np.arange(-4, -1)
    )

    clf = GridSearchCV(svm, param_grid)
    clf.fit(X_train, y_train)

    clf.score(X_test, y_test)
    print(f"Best parameters: {clf.best_params_}, best value: {clf.best_score_}")

    # TODO Use the parameters you have found to instantiate a LinearSVM.
    # the `fit` method returns a list of scores that you should plot in order
    # to monitor the convergence. When does the classifier converge?
    svm = LinearSVM(C=clf.best_params_["C"], eta=clf.best_params_["eta"])
    scores = svm.fit(X_train, y_train)
    plt.plot(scores)
    plt.show()
    test_score = svm.score(X_test, y_test)
    print(f"Test Score: {test_score}")

    # TODO plot the decision boundary!
    plt.figure()
    plotting.plot_decision_boundary(X_train, svm)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.show()
