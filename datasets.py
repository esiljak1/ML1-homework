import pickle as pkl

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_toy_dataset(dataset_idx, random_state=42, apply_noise=False, remove_outlier=False):
  with open('data/dataset%d.pkl' % dataset_idx, 'rb') as f:
    X, y = pkl.load(f)

  if remove_outlier:
    assert dataset_idx == 1, "Only dataset 1 is supported."
    mask = np.arange(X.shape[0]) != 50
    X = X[mask]
    y = y[mask]
     
  if apply_noise:
    rng = np.random.default_rng(42)
    X += rng.normal(size=X.shape, scale=0.06)

  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)
  return X_train, X_test, y_train, y_test


def get_heart_dataset(random_state=42):
  data = np.loadtxt('data/heart.csv', delimiter=',', skiprows=1).astype(np.float)
  with open('data/heart.csv', 'r') as f:
    labels = f.readline()
  labels = np.array(labels.split(',')[:13])
  X, y = data[:, :13], data[:, 13]
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  return X_train, X_test, y_train, y_test, labels
