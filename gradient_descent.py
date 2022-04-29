import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_eggholder_function(f):
    '''
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    '''
    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f([XX, YY])

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the max number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: function representing the gradient of f
    :param x: vector, initial point
    :param learning_rate:
    :param max_iter: maximum number of iterations
    :return: x (solution, vector), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)
    # Implement the gradient descent algorithm
    # E_list should be appended in each iteration, with the current value of the cost

    for i in range(max_iter):
        x = x - learning_rate * df(x)
        E_list[i] = f(x)

    return x, E_list


def eggholder(x):
    # TODO: Implement the cost function specified in the HW1 sheet
    # x[0]=x and x[1]=y in Eq 1
    # z = np.sin(x[1])+x[0]
    z = -(x[1]+47) * np.sin((abs(x[0]/2+x[1]+47))**0.5) - x[0] * np.sin((abs(x[0]-(x[1]+47)))** 0.5)
    return z


def gradient_eggholder(x):
    # TODO: Implement gradients of the Eggholder function w.r.t. x and y
    # x[0]=x and x[1]=y in Eq 1
    # problematic points
    if x[1] == -x[0] / 2 - 47 or x[1] == x[0] - 47:
        return [np.inf, np.inf]

    grad_x = -(x[1] + 47) * np.cos((abs(x[0]/2 + x[1] + 47)) ** 0.5) * 1/2 * 1/((abs(x[0]/2 + x[1] + 47)) ** 0.5) * (x[0]/2 + x[1] + 47)/abs(x[0]/2 + x[1] + 47) * 1/2 \
             - np.sin((abs(x[0] - (x[1] + 47))) ** 0.5) \
             - x[0] * np.cos((abs(x[0] - (x[1] + 47))) ** 0.5) * 1/2 * 1/((abs(x[0] - (x[1] + 47))) ** 0.5) * (x[0] - (x[1] + 47))/abs(x[0] - (x[1] + 47))

    grad_y = - np.sin((abs(x[0]/2 + x[1] + 47)) ** 0.5)\
             - (x[1] + 47) * np.cos((abs(x[0]/2 + x[1] + 47)) ** 0.5) * 1/2 * 1/((abs(x[0]/2 + x[1] + 47)) ** 0.5) * (x[0]/2 + x[1] + 47)/abs(x[0]/2 + x[1] + 47) \
             - x[0] * np.cos((abs(x[0] - (x[1] + 47))) ** 0.5) * 1/2 * 1/((abs(x[0] - (x[1] + 47))) ** 0.5) * (x[0] - (x[1] + 47))/abs(x[0] - (x[1] + 47)) * (-1)

    return np.array([grad_x, grad_y])
