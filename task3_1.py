import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import plotting
from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # TODO start with `n_estimators = 1`
    rf = RandomForestClassifier()
    clf = ...
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)
    #TODO plot decision boundary
