import matplotlib.pyplot as plt

import plotting
from datasets import get_toy_dataset
from task2_1 import LinearSVM
from sklearn.model_selection import GridSearchCV
import numpy as np

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(1, remove_outlier=True)
    svm = LinearSVM()
    # TODO use grid search to find suitable parameters!

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
