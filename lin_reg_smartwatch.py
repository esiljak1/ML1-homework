import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv


def pearson_coefficient(x, y):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional)
    :return: Pearson coefficient of correlation
    """
    # Implement it yourself, you are allowed to use np.mean, np.sqrt, np.sum.
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.dot(x, y)
    sum_x2 = np.dot(x, x)
    sum_y2 = np.dot(y, y)
    n = len(x)

    r = (n * sum_xy - sum_x * sum_y) / (np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)))
    return r


def fit_predict_mse(x, y):
    """
    :param x: Variable 1 (Feature vector (one-dimensional))
    :param y: Variable_2 (one-dimensional), dependent variable
    :return: theta_star - optimal parameters found; mse - Mean Squared Error
    """
    m = np.size(x)
    x_new = np.array([x])
    ones = np.array([np.ones(m)])
    X = np.concatenate((ones.T, x_new.T), axis =1)  # TODO create a design matrix

    pseudo_inv = pinv(X)
    theta_star = pseudo_inv.dot(y)  # TODO calculate theta using pinv from numpy.linalg (already imported)

    y_pred = X.dot(theta_star)  # TODO predict the value of y
    err_sq = (y_pred - y) ** 2
    mse = np.mean(err_sq)  # TODO calculate MSE

    return theta_star, mse


def scatterplot_and_line(x, y, theta, xlabel='x', ylabel='y', title='Title'):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # theta will be an array with two coefficients, representing the slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.
    # TODO
    plt.scatter(x, y)  # plot data points
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # we plot a line y = ax + b. theta = [b, a]
    x_line = np.arange(max(x))
    y_line = theta[1]*x_line+theta[0]
    plt.plot(x_line, y_line, "r")
    plt.show()




