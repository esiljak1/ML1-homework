import random as rd
import numpy as np
from lin_reg_ecg import test_fit_line, find_new_peak, check_if_improved, bonus_plot
from lin_reg_smartwatch import pearson_coefficient, fit_predict_mse, scatterplot_and_line
from gradient_descent import eggholder, gradient_eggholder, gradient_descent, plot_eggholder_function
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cm_blue_orange = ListedColormap(['blue', 'orange'])


def task_1_1():
    print('---- Task 1.1 ----')
    test_fit_line()

    # Load ecg signal from 'data/ecg.npy' using np.load
    ecg = np.load("./data/ecg.npy")

    # Load indices of peaks from 'indices_peaks.npy' using np.load. There are 83 peaks.
    peaks = np.load("./data/indices_peaks.npy")

    # Create a "timeline". The ecg signal was sampled at sampling rate of 180 Hz, and in total 50 seconds.
    # Datapoints are evenly spaced. Hint: shape of time signal should be the same as the shape of ecg signal.
    time = np.arange(0, 50, 1/180)

    print(f'time shape: {time.shape}, ecg signal shape: {ecg.shape}')
    print(f'First peak: ({time[peaks[0]]:.3f}, {ecg[peaks[0]]:.3f})')

    # Plot of ecg signal (should be similar to the plot in Fig. 1A of HW1, but shown for 50s, not 8s)
    plt.plot(time, ecg)
    plt.plot(time[peaks], ecg[peaks], "x")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [V]')
    plt.show()

    new_peaks = np.zeros(peaks.size)
    new_sig = np.zeros(peaks.size)
    improved = np.zeros(peaks.size)

    for i, peak in enumerate(peaks):
        x_new, y_new = find_new_peak(peak, time, ecg)
        new_peaks[i] = x_new
        new_sig[i] = y_new
        improved[i] = check_if_improved(x_new, y_new, peak, time, ecg)

    print(f'Improved peaks: {np.sum(improved)}, total peaks: {peaks.size}')
    print(f'Percentage of peaks improved: {np.sum(improved) / peaks.size :.4f}')

    bonus_plot(time, ecg, peaks, (0, 8.2))
    bonus_plot(time, ecg, peaks, (0.05, 0.25))


def task_1_2():
    print('\n---- Task 1.2 ----')
    # hours_sleep, hours_work, avg_pulse, max_pulse, duration, exercise_intensity, fitness_level, calories
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3,
                    "duration": 4, "exercise_intensity": 5,
                    "fitness_level": 6, "calories": 7}

    # Load the data from 'data/smartwatch_data.npy' using np.load
    smartwatch_data = np.load("data/smartwatch_data.npy")
    # Now you can access it, for example,  smartwatch_data[:, column_to_id["hours_sleep"]]
    # Meaningful relations
    meaningful_relations = [("duration", "fitness_level"), ("fitness_level", "calories"), ("duration", "calories")]

    for i in range(3):
        first_column_name = meaningful_relations[i][0]
        second_column_name = meaningful_relations[i][1]
        x = smartwatch_data[:, column_to_id[first_column_name]]
        y = smartwatch_data[:, column_to_id[second_column_name]]
        theta, mse = fit_predict_mse(x, y)

        scatterplot_and_line(x, y, theta, xlabel=first_column_name, ylabel=second_column_name, title=f'{first_column_name} - {second_column_name} relation')

        # TODO (use fit_predict_mse)
        corr = pearson_coefficient(x, y)
        print(f'Pearson coeff: {corr}, theta={theta}, MSE={mse}')

    # No linear relations
    # TODO (use fit_predict_mse)
    no_relations = [("hours_sleep", "avg_pulse"), ("max_pulse", "fitness_level"), ("max_pulse", "duration")]
    for i in range(3):
        first_column_name = no_relations[i][0]
        second_column_name = no_relations[i][1]
        x = smartwatch_data[:, column_to_id[first_column_name]]
        y = smartwatch_data[:, column_to_id[second_column_name]]
        theta, mse = fit_predict_mse(x, y)

        scatterplot_and_line(x, y, theta, xlabel=first_column_name, ylabel=second_column_name,
                             title=f'{first_column_name} - {second_column_name} relation')

        # TODO (use fit_predict_mse)
        corr = pearson_coefficient(x, y)
        print(f'Pearson coeff: {corr}, theta={theta}, MSE={mse}')


def task_2():
    print('\n---- Task 2 ----')

    def plot_datapoints(X, y, title, fig_name='fig.png'):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        fig.suptitle(title, y=0.93)

        p = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_blue_orange)

        axs.set_xlabel('x1')
        axs.set_ylabel('x2')
        axs.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(0.96, 1.15))

        fig.savefig(fig_name) # Uncomment if you want to save it

    # Load the data from 'data/X.npy' using np.load
    X_original = np.load("./data/X.npy") # TODO: change me
    X = []
    y = []

    for task in [0, 1, 2]:
        print(f'---- Logistic regression task {task + 1} ----')
        if task == 0:
            # Load the data from 'data/targets-dataset-1.npy' using np.load
            y = np.load("./data/targets-dataset-1.npy")
            additional_column = np.ones((X_original.shape[0], 1))
            X = np.hstack((X_original, additional_column))  # TODO # create design matrix based on features X_original
            for i in range(X.shape[0]):
                x1 = X[i, 0]
                x2 = X[i, 1]
                if x1 < 10 or x2 < 10:
                    X[i, 2] = 0

        elif task == 1:
            # Load the data from 'data/targets-dataset-2.npy' using np.load
            y = np.load("./data/targets-dataset-2.npy")
            additional_column = np.ones((X_original.shape[0], 1))
            X = np.hstack((X_original, additional_column))  # TODO # create design matrix based on features X_original
            for i in range(X.shape[0]):
                x1 = X[i, 0]
                x2 = X[i, 1]
                if x1 < 10 or x2 > 20:
                    X[i, 2] = 0
        elif task == 2: 
            # Load the data from 'data/targets-dataset-3.npy' using np.load
            y = np.load("./data/targets-dataset-3.npy")

            additional_column = np.zeros((X_original.shape[0], 1))
            X = np.hstack((X_original, additional_column))  # TODO # create design matrix based on features X_original
            for i in range(X.shape[0]):
                x1 = X[i, 0]
                x2 = X[i, 1]
                if x1 ** 2 + x2 ** 2 + 24 < 625:
                    X[i, 2] = 1

        # plot_datapoints(X, y, 'Targets', './plots/targets_' + str(task+1) + '.png')

        # Spilit data into train and test sets, using train_test_split function that is already imported 
        # We want 20% of the data to be in the test set. Fix the random_state parameter (use value 0)).
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Create a classifier, and fit the model to data
        clf = LogisticRegression(penalty='l2')
        clf.fit(X_train, y_train)
        
        acc_train = clf.score(X_train, y_train)
        acc_test = clf.score(X_test, y_test)
        print(f'Train accuracy: {acc_train:.4f}. Test accuracy: {acc_test:.4f}.')
        
        # TODO: Calculate the loss.
        # Calculate PROBABILITIES of predictions. Output will be with the second dimension that equals 2, because we have 2 classes. 
        # (The returned estimates for all classes are ordered by the label of classes.)
        # When calculating log_loss, provide yhat_train and yhat_test of dimension (n_samples, ). That means, "reduce" the dimension, 
        # simply by selecting (indexing) the probabilities of the positive class. 

        prediction_probabilities_test = clf.predict_proba(X_test)[:, 1]
        prediction_probabilities_train = clf.predict_proba(X_train)[:, 1]
        loss_train, loss_test = log_loss(y_train, prediction_probabilities_train), log_loss(y_test, prediction_probabilities_test) # TODO use log_loss from sklearn.metrics (already imported)
        print(f'Train loss: {loss_train:.4f}. Test loss: {loss_test:.4f}.')

        # Calculate predictions, we need them for the plots
        yhat_train = prediction_probabilities_train  # TODO
        yhat_test = prediction_probabilities_test # TODO
        yhat = clf.predict_proba(X)[:, 1]  # TODO and use the whole dataset

        plot_datapoints(X_train, yhat_train, 'Predictions on the train set', fig_name='./plots/Result_train/logreg_train' + str(task + 1) + '.png')
        plot_datapoints(X_test, yhat_test, 'Predictions on the test set', fig_name='./plots/Result_test/logreg_test' + str(task + 1) + '.png')
        plot_datapoints(X, yhat, 'Predictions on the whole set', fig_name='./plots/Result_overall/logreg_whole_ds' + str(task + 1) + '.png')

        # TODO: Print theta vector (and also the bias term). Hint: check Attributes of the classifier
        print(f'Theta-{task + 1}: {clf.coef_}')
        print(f'Bias-{task + 1}: {clf.intercept_}')


def task_3():
    print('\n---- Task 3 ----')
    # Plot the function, to see how it looks like
    plot_eggholder_function(eggholder)

    # TODO: choose a 2D random point from randint (-512, 512)
    x0 = np.array([rd.uniform(-512, 512), rd.uniform(-512, 512)])
    print(f'Starting point: x={x0}')

    max_iter = [1000, 2000, 5000]
    learning_rate = [0.02, 0.05, 0.3]
    convergence_names = ['slow', 'moderate', 'fast']

    for i in range(3):
        # Call the function gradient_descent. Choose max_iter, learning_rate.
        x, E_list = gradient_descent(eggholder, gradient_eggholder, x0, learning_rate[i], max_iter[i])
        print(f'Minimum found: f({x}) = {eggholder(x)}')

        # TODO Make a plot of the cost over iteration. Do not forget to label the plot (xlabel, ylabel, title).
        plt.xlabel("Iteration")
        plt.ylabel("Cost value")
        plt.title(f'Plot of the {convergence_names[i]} convergence')
        x = np.arange(max_iter[i])
        y = E_list[x]
        plt.plot(x, y)
        plt.show()

    x_min = np.array([512, 404.2319])
    print(f'Global minimum: f({x_min}) = {eggholder(x_min)}')

    # Test 1 - Problematic point 1. See HW1, Tasks 3.6 and 3.7.
    x, y = 0, -47  # TODO: change me
    print('A problematic point: ', gradient_eggholder([x, y]))

    # Test 2 - Problematic point 2. See HW1, Tasks 3.6 and 3.7.
    x, y = 47, 0  # TODO: change me
    print('Another problematic point: ', gradient_eggholder([x, y]))


def main():
    task_1_1()
    task_1_2()
    task_2()
    task_3()


if __name__ == '__main__':
    main()
