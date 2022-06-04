import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, cross_val_score

from datasets import get_toy_dataset

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_toy_dataset(4)

    #TODO fit a random forest classifier and """"check how well it performs on the test set after tuning the parameters""",
    # report your results
    rf = RandomForestClassifier(criterion="entropy", max_depth=14, random_state=20)
    """
    grid_params = dict(
      n_estimators=[10, 50, 100, 200],
      criterion=["gini", "entropy"],
      max_depth=[1, 5, 10, None],
      max_leaf_nodes=[5, 10, None]
    )
    clf = GridSearchCV(rf, grid_params)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    """
    rf.fit(X_train, y_train)
    test_score = rf.score(X_test, y_test)
    print(f"Test Score RFC: {test_score}")


    #TODO fit a SVC and """find suitable parameters"""", report your results
    svc = SVC(C=1, kernel="rbf", gamma="scale")
    """
    grid_params = dict(
      C=10.**np.arange(0, 2),
      kernel=["poly", "rbf"],
      gamma=["scale", "auto"]
    )
    
    clf = GridSearchCV(svc, grid_params)
    clf.fit(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
    clf.score(X_test, y_test)
    print(f"Best parameters: {clf.best_params_}, best score svc: {clf.best_score_}")
    """
    svc.fit(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
    test_score = svc.score(X_test, y_test)
    print(f"Test score svc: {test_score}")

    # TODO create a bar plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)
    # of the `feature_importances_` of the RF classifier.

    plt.figure()
    plt.barh([f"Feature {i}" for i in range(len(rf.feature_importances_))], rf.feature_importances_)
    plt.xlabel("Feature importance")
    plt.ylabel("Features")
    plt.title("Importance of the feature")
    plt.show()

    # TODO create another RF classifier
    # Use recursive feature elimination (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)
    # to automatically choose the best number of parameters
    # set `scoring = 'accuracy'` to look for the feature subset with highest accuracy
    # and fit the RFECV to the training data
    rf = RandomForestClassifier(criterion="entropy", max_depth=14, random_state=20)
    rfecv = RFECV(rf, scoring="accuracy")
    rfecv.fit(X_train, y_train)

    # TODO use the RFECV to transform the training and test dataset -- it automatically removes the least important
    # feature columns from the datasets. You don't have to change y_train or y_test
    # Fit a SVC classifier on the new dataset. Do you see a difference in performance?
    X_train = rfecv.transform(X_train)
    X_test = rfecv.transform(X_test)
    svc = SVC(C=1, kernel="rbf", gamma="scale")
    svc.fit(np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
    test_score_svc = svc.score(X_test, y_test)
    cross_val = cross_val_score(svc, X_test, y_test)
    print(f"SVC with applied RFECV test score: {test_score_svc}, cross_val_score: {np.mean(cross_val)}")
