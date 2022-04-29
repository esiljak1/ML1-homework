import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt


def test_fit_line():
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 4, 5, 6])
    a, b = _fit_line(x, y)

    print(a, b)  # Should be: a = 1.0, b = 3.0
    # TODO BONUS task - write an assert command that checks a and b (and what it should be in this test case),
    eps = 1e-4
    assert abs(a - 1) <= eps, "a should be 1"
    assert abs(b - 3) <= eps, "b should be 3"
    # and a message to be displayed if the test fails.


def _fit_line(x, y):
    """
    :param x: x coordinates of data points
    :param y: y coordinates of data points
    :return: a, b - slope and intercept of the fitted line
    """
    
    # TODO BONUS task - write an assert command to check if there are at least two data points given, and a message to be displayed if the test fails.
    assert len(x) >= 2 and len(y) >= 2, "At least two data points must be provided to this function"

    # TODO calculate a and b (either in the form of sums, or by using a design matrix and pinv from numpy.linalg (already imported).
    matrix_x = np.asmatrix(x).T
    design_matrix = np.hstack((np.ones((x.shape[0], 1)), matrix_x))
    theta = pinv(design_matrix).dot(y)

    return theta[0, 1], theta[0, 0]


def _intersection(a, b, c, d):
    """
    :param a: slope of the "left" line
    :param b: intercept of the "left" line
    :param c: slope of the "right" line
    :param d: intercept of the "right" line
    :return: x, y - corrdinates of the intersection of two lines
    """
    x = 0 if a-c == 0 else (d-b)/(a-c)  # TODO x-coordinate of the intersection
    y = a*x + b  # TODO y-coordinate of the intersection
    return x, y


def check_if_improved(x_new, y_new, peak, time, signal):
    """
    :param x_new: x-coordiinate of a new peak
    :param y_new: y-coordinate of a new peak
    :param peak: index of the peak that we were improving
    :param time: all x-coordinates for ecg signal
    :param signal: all y-coordinates of signal (i.e., ecg signal)
    :return: 1 - if new peak is improvment of the old peak, otherwise 0
    """

    if y_new > signal[peak] and time[peak-1] < x_new < time[peak + 1]:
        return 1
    return 0


def find_new_peak(peak, time, sig):
    """
    This function fits a line through points left of the peak, then another line through points right of the peak.
    Once the coefficients of both lines are obtained, the intersection point can be calculated, representing a new peak.

    :param peak: Index of the peak
    :param time: Time signal (the whole signal, 50 s)
    :param sig: ECG signal (the whole signal for 50 s)
    :return:
    """
    # left line
    n_points = 2  # TODO choose the number of points for the left line
    ind = 0  # TODO indices for the left line, choose if you want to include the peak or not)
    x = time[peak - n_points: peak - ind]  # TODO
    y = sig[peak - n_points: peak - ind]  # TODO
    a, b = _fit_line(x, y)

    # right line
    n_points = 3  # TODO choose the number of points for the right line
    ind = 1  # TODO indices for the right line, choose if you want to include the peak or not
    x = time[peak + 1 - ind: peak + 1 + n_points]  # TODO
    y = sig[peak + 1 - ind: peak + 1 + n_points]  # TODO
    c, d = _fit_line(x, y)

    # find intersection point
    x_new, y_new = _intersection(a, b, c, d)
    return x_new, y_new


def bonus_plot(time, ecg, peaks, xlim):
    for i, peak in enumerate(peaks):
        if time[i] > xlim[1]:
            break
        x = time[peak - 2: peak]
        y = ecg[peak - 2: peak]
        a, b = _fit_line(x, y)
        x_plot = np.linspace(xlim[0], xlim[1], 100)
        y_plot = a * x_plot + b
        plt.plot(x_plot, y_plot, "b")

        x = time[peak: peak + 4]
        y = ecg[peak: peak + 4]
        c, d = _fit_line(x, y)
        x_plot = np.linspace(xlim[0], xlim[1], 100)
        y_plot = c * x_plot + d
        plt.plot(x_plot, y_plot, "b")

        x_new, y_new = _intersection(a, b, c, d)
        plt.plot(x_new, y_new, marker="o", color="#000000")

    plt.plot(time, ecg, color="#cf6223")
    plt.plot(time[peaks], ecg[peaks], "gx")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.xlim(xlim)
    plt.ylim((-2, 3))
    plt.show()
