import matplotlib.pyplot as plt
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
    k_range = list(range(1, 100))
    grid_dict = dict(k = k_range)
    clf = GridSearchCV(knn, grid_dict, return_train_score=True)
    clf.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print(f"Test Score: {test_score}")
    print(f"Dataset {idx}: {clf.best_params_}")

    plt.figure()
    plotting.plot_decision_boundary(X_train, clf)
    plotting.plot_dataset(X_train, X_test, y_train, y_test)
    # TODO you8 should use the plt.savefig(...) function to store your plots before calling plt.show()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Datasets and decision boundary")
    plt.savefig("figure1_task_1.2")
    plt.show()

    # Training and Test accuracies
    plt.figure()
    plt.plot(range(1,100), clf.cv_results_['mean_train_score'], label = "Training accuracy") # He añadido el range para que empiece en k=1
    plt.plot(range(1,100), clf.cv_results_['mean_test_score'], color = "r", label = "Test accuracy")
    plt.legend() # Cuando consiga correr el programa revisar esto :)
    plt.xlabel("k")
    plt.ylabel("Accuracies")
    plt.title("Training and test accuracy")
    plt.savefig("figure2_task_1.2")
    plt.show()

