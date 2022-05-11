import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    pca = PCA(n_components)  # TODO # Create an instance of PCA from sklearn.decomposition (already imported).

    X_reduced = pca.fit_transform(features)  # TODO # Fit the model with features, and apply the transformation on the features.

    explained_var = sum(pca.explained_variance_ratio_)  # TODO: # Calculate the percentage of variance explained
    print(f'Explained variance: {explained_var}')
    return X_reduced


def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [5, 100, 200]  # TODO create a list

    for n_hid in n_hidden_neurons:
        # TODO create an instance of MLPClassifier from sklearn.neural_network (already imported).
        # Set parameters (some of them are specified in the HW2 sheet).
        clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=n_hid).fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)  # TODO
        test_acc = clf.score(X_test, y_test)  # TODO
        loss = clf.loss_  # TODO
        print(f'Data for n_hid={n_hid}')
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}\n')


def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [5, 100, 200]  # TODO create a list

    for n_hid in n_hidden_neurons:
        # TODO create an instance of MLPClassifier from sklearn.neural_network (already imported).
        # Set parameters (some of them are specified in the HW2 sheet).
        clf = MLPClassifier(random_state=0, max_iter=500, hidden_layer_sizes=n_hid, alpha = 1).fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)  # TODO
        test_acc = clf.score(X_test, y_test)  # TODO
        loss = clf.loss_  # TODO
        print(f'Data for n_hid={n_hid}')
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}\n')


def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print (mean +/- std) accuracy on the training and test set.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [0, 2, 27, 55, 100] # TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    # TODO create an instance of MLPClassifier, check the perfomance for different seeds
    # We choose n_hid = 200

    i = 0

    for seed in seeds:
        np.random.seed(seed)
        clf = MLPClassifier(random_state=seed, max_iter=500, hidden_layer_sizes=200, alpha = 1).fit(X_train, y_train)
        train_acc = clf.score(X_train, y_train)  # TODO
        test_acc = clf.score(X_test, y_test)  # TODO for each seed
        loss = clf.loss_  # TODO for each seed
        print(f'Data for random_state={seed}')
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}\n')

        train_acc_arr[i] = train_acc
        test_acc_arr[i] = test_acc
        i = i+1

    train_acc_mean = np.average(train_acc_arr) # TODO
    train_acc_std = np.std(test_acc_arr) # TODO
    test_acc_mean = np.average(test_acc_arr) # TODO
    test_acc_std = np.std(test_acc_arr) # TODO

    # TODO: print min and max accuracy as well
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')

    # Task 1.5
    # We plot loss curve for seed = 27. alpha = 1
    print("----- Task 1.5 -----")
    clf_new = MLPClassifier(random_state=27, max_iter=500, hidden_layer_sizes=200, alpha = 1).fit(X_train, y_train)
    x_line = range(0, len(clf_new.loss_curve_))
    y_line = clf_new.loss_curve_
    plt.plot(x_line, y_line)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    plt.title("Loss over iteration. Seed = 27.")
    plt.show()

    # Task 1.6
    print("----- Task 1.6 -----")
    # TODO: Confusion matrix and classification report (for one classifier that performs well)
    print("Predicting on the test set")
    y_pred = clf_new.predict(X_test) # TODO calculate predictions
    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):
    """
    BONUS task: Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    parameters = {
        'alpha': [0.0, 0.001, 1.0],
        'activation': ['identity', 'logistic', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'hidden_layer_sizes': [(100,), (200,)]
    } # TODO create a dictionary of params

    nn = MLPClassifier(max_iter=500, random_state=0, early_stopping=True)  # TODO create an instance of MLPClassifier. Do not forget to set parameters as specified in the HW2 sheet.
    grid_search = sklearn.model_selection.GridSearchCV(nn, parameters, n_jobs=-1)  # TODO create an instance of GridSearchCV from sklearn.model_selection (already imported) with
    # appropriate params. Set: n_jobs=-1, this is another parameter of GridSearchCV, in order to get faster execution of the code.

    # TODO call fit on the train data
    grid_search.fit(X_train, y_train)
    # TODO print the best score
    print(f'Best score: {grid_search.best_score_}')
    # TODO print the best parameters found by grid_search
    print(f'Parameters for the best score: {grid_search.best_params_}')
