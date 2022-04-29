import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    # pca = # TODO # Create an instance of PCA from sklearn.decomposition (already imported). 

    X_reduced = np.zeros((features.shape[0], n_components)) # TODO # Fit the model with features, and apply the transformation on the features.

    explained_var = 0 # TODO: # Calculate the percentage of variance explained
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

    n_hidden_neurons = 2 # TODO create a list
    # TODO create an instance of MLPClassifier from sklearn.neural_network (already imported).
    # Set parameters (some of them are specified in the HW2 sheet).

    train_acc = 0 # TODO
    test_acc = 0 # TODO
    loss = 0 # TODO
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')

def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # Copy your code from train_nn, but experiment now with regularization (alpha, early_stopping).

    train_acc = 0 # TODO
    test_acc =  0 # TODO
    loss =  0 # TODO
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')


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
    seeds = [0] # TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    # TODO create an instance of MLPClassifier, check the perfomance for different seeds

    train_acc = 0 # TODO 
    test_acc =  0 # TODO for each seed
    loss =  0 # TODO for each seed
    print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
    print(f'Loss: {loss:.4f}')


    train_acc_mean = 0 # TODO
    train_acc_std = 0 # TODO
    test_acc_mean = 0 # TODO
    test_acc_std = 0 # TODO
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')
    # TODO: print min and max accuracy as well


    # TODO: Confusion matrix and classification report (for one classifier that performs well)
    print("Predicting on the test set")
    # y_pred = 0 # TODO calculate predictions
    # print(classification_report(y_test, y_pred)) 
    # print(confusion_matrix(y_test, y_pred, labels=range(10)))


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
    parameters = None # TODO create a dictionary of params

    # nn = # TODO create an instance of MLPClassifier. Do not forget to set parameters as specified in the HW2 sheet.
    # grid_search = # TODO create an instance of GridSearchCV from sklearn.model_selection (already imported) with
    # appropriate params. Set: n_jobs=-1, this is another parameter of GridSearchCV, in order to get faster execution of the code.

    # TODO call fit on the train data
    # TODO print the best score
    # TODO print the best parameters found by grid_search
