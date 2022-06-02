import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
    for idx in [1, 2, 3]:
        X_train, X_test, y_train, y_test = get_toy_dataset(idx)
        svc = SVC(tol=1e-4)
        # TODO perform grid search, decide on suitable parameter ranges and state sensible parameter ranges in your report
        # need to update the values of C and gamma
        grid_dict = dict(
          C=10**np.arange(-2.0, 2.0),
          gamma=10**np.arange(-4.0, 2.0),
          kernel=["linear", "rbf"]
        )
        clf = GridSearchCV(svc, grid_dict)
        clf.fit(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        print("Test Score:", test_score)
        print("Mean cross-validated Test Score:", clf.best_score_)
        print(f"Dataset {idx}: {clf.best_params_}")


        # TODO plot and save decision boundaries
        plt.figure()
        plotting.plot_decision_boundary(X_train, clf)
        plotting.plot_dataset(X_train, X_test, y_train, y_test)
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title(f"Dataset {idx} (train and test) and decision boundary")
        plt.show()
