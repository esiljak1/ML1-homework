import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from datasets import get_toy_dataset

if __name__ == '__main__':
  X_train, X_test, y_train, y_test = get_toy_dataset(4)

  #TODO fit a random forest classifier and """"check how well it performs on the test set after tuning the parameters""",
  # report your results
  rf = RandomForestClassifier()
  rf.fit(X_train, y_train)

  #TODO fit a SVC and """find suitable parameters"""", report your results
  svc = SVC()
  X = np.append(X_train, X_test)
  y = np.append(y_train, y_test)
  svc.fit(X,y)

  # TODO create a bar plot (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html#matplotlib.pyplot.barh)
  # of the `feature_importances_` of the RF classifier.

  plt.barh(rf.feature_importances_)
  # FALTAN LOSDETALLES DEL PLOT!!!!

  # TODO create another RF classifier
  # Use recursive feature elimination (https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV)
  # to automatically choose the best number of parameters
  # set `scoring = 'accuracy'` to look for the feature subset with highest accuracy
  # and fit the RFECV to the training data
  rf = ...
  rfecv = ...

  # TODO use the RFECV to transform the training and test dataset -- it automatically removes the least important
  # feature columns from the datasets. You don't have to change y_train or y_test
  # Fit a SVC classifier on the new dataset. Do you see a difference in performance?
  svc = ...
