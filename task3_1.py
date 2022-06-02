from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from datasets import get_toy_dataset

if __name__ == '__main__':
  for idx in [1, 2, 3]:
    X_train, X_test, y_train, y_test = get_toy_dataset(idx)
    # TODO start with `n_estimators = 1`
    dict_parameters = dict( n_estimators = 1, max_depth= range(1,100))
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, dict_parameters)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)

    # TODO plot decision boundary
    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test) # Plotear los dos data sets o solo el test data set???
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Decision boundary")
    plt.savefig("figure1_task_3.1")
    plt.show()

    # TODO start with `n_estimators = 100
    dict_parameters = dict(n_estimators=100, max_depth=range(1, 100)) # Como variamos el max depth??
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, dict_parameters)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Dataset {idx}: {clf.best_params_}")
    print("Test Score:", test_score)

    # TODO plot decision boundary
    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)  # Plotear los dos data sets o solo el test data set???
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Decision boundary n_estimators = 100, data set x") # preguntarle a emin como poner dta sett 1, 2, 3
    plt.savefig("figure2_task_3.1")
    plt.show()