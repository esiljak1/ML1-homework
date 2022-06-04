import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import plotting
from datasets import get_toy_dataset
from task1_1 import KNearestNeighborsClassifier

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(2, apply_noise=True)
    for k in [1, 5, 20, 50, 100]:
        # TODO fit your KNearestNeighborsClassifier with k in {1, 30, 100} and plot the decision boundaries
        clf = KNearestNeighborsClassifier(k=k)
        clf.fit(X_train, y_train)
        # TODO you can use the `cross_val_score` method to manually perform cross-validation
        # TODO report mean cross-validated scores!
        cross_val = cross_val_score(clf, X_test, y_test)
        test_score = clf.score(X_test, y_test)
        print(f"Test Score for k={k}: {test_score}")
        print(f"Mean cross validation score for k={k}: {np.mean(cross_val)}\n")
        # TODO plot the decision boundaries!
        plotting.plot_decision_boundary(X_test, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.xlabel("X")
        plt.ylabel ("y")
        plt.title(f"Data set 2 and decision boundary for k ={k}")
        plt.show()

    # TODO find the best parameters for the noisy dataset!
    knn = KNearestNeighborsClassifier()
    k = range(1, 101)
    param_grid = dict(
      k=k
    )
    clf = GridSearchCV(clf, param_grid, return_train_score=True)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    print(f"Best parameters: {clf.best_params_}, best score: {clf.best_score_}")
    # TODO The `cv_results_` attribute of `GridSearchCV` contains useful aggregate information
    # such as the `mean_train_score` and `mean_test_score`. Plot these values as a function of `k` and report the best
    # parameters. Is the classifier very sensitive to the choice of k?
    mean_train_score = clf.cv_results_["mean_train_score"]
    mean_test_score = clf.cv_results_["mean_test_score"]

    plt.plot(k, mean_train_score)
    plt.suptitle("Mean train score")
    plt.xlabel("K")
    plt.ylabel("Mean score")
    plt.show()

    plt.plot(k, mean_test_score)
    plt.suptitle("Mean test score")
    plt.xlabel("K")
    plt.ylabel("Mean score")
    plt.show()

