import numpy as np


def shuffle_dataset(X: np.array, y: np.array):
    # perm = np.random.permutation(X.shape[0])
    # return X[perm], y[perm]
    return X, y


def empty_array_from_template(template_array):
    return np.ndarray((0,) + template_array.shape[1:])
