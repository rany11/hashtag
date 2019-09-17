import numpy as np

from src.backend.active_learning.committee_utils.committee import Committee
from src.experimentation.dataset.utils import shuffle_dataset


class CommitteeSplitData(Committee):
    def train(self):
        return self.__split_data_training()

    def __split_data_training(self):
        self.classifiers = []
        X_train, y_train = shuffle_dataset(self.X_train, self.y_train)
        for i, X, y in zip(range(self.target_committee_size),
                           np.array_split(X_train, self.target_committee_size),
                           np.array_split(y_train, self.target_committee_size)):
            self.classifiers.append(
                self.base_estimator.fit(
                    np.concatenate((self.X_all_labels, X)),
                    np.concatenate((self.y_all_labels, y))
                )
            )
        return
