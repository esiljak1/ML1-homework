from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    mse = mean_squared_error(targets, predictions)  # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    return mse


def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons_list = [10,100,200] # TODO (try at least 3 different numbers of neurons)

    # TODO: MLPRegressor, choose the model yourself
    # We are going to use function GridSearchCV
    parameters = {
                    "hidden_layer_sizes" : n_hidden_neurons_list,               # number of neurons
                    "early_stopping" : [True, False],                           # Regularization
                    "learning_rate" : ["constant", "invscaling", "adaptive"],   # optimizer
                    "max_iter" : [100, 200, 500]
                     }
    clf = GridSearchCV(MLPRegressor(), parameters).fit(X_train, y_train)
    print (clf.best_params_)


    """"
    # Calculate predictions
    y_pred_train = 0 # TODO
    y_pred_test = 0 # TODO
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')
    """