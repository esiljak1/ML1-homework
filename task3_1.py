import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # TODO start with `n_estimators = 1`
    dict_parameters = dict( n_estimators = [1], max_depth= range(1,101))
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, dict_parameters)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)
    print(f"Mean cross validation score for k={idx}: {clf.best_score_}\n")

    # TODO plot decision boundary
    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"Decision boundary n_estimators = 1. Data set {idx}")
    plt.show()

    # TODO start with `n_estimators = 100
    dict_parameters = dict(n_estimators=[100], max_depth=range(1, 100))
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, dict_parameters)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)
    print(f"Mean cross validation score: {clf.best_score_}\n")

    # TODO plot decision boundary
    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title(f"Decision boundary n_estimators = 100. Data set {idx}")
    plt.show()