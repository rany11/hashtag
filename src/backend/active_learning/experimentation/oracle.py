import numpy as np


class Oracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        return

    def query_update(self, idx):
        X_size = self.X.shape[0]
        non_queried_idx = np.setdiff1d(np.arange(X_size), idx)

        samples = self.X[idx, :]
        labels = self.y[idx]

        self.X = self.X[non_queried_idx, :]
        self.y = self.y[non_queried_idx]
        return samples, labels

    def nb_left_samples(self):
        return self.X.shape[0]

    def is_out_of_data(self):
        return self.nb_left_samples() == 0
