from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class Committee(ABC):
    @staticmethod
    def majority(votes):
        return np.apply_along_axis(lambda V: Counter(V).most_common()[0][0],
                                   0,
                                   votes)

    @staticmethod
    def probability_avr(votes_prob):
        return np.apply_along_axis(np.average, 0, votes_prob)

    @staticmethod
    def votes_to_onehot(votes, all_labels):
        encoder = OneHotEncoder(sparse=False, categories=[all_labels])
        encoder.fit(votes.reshape(-1, 1))
        onehot_votes = np.stack([encoder.transform(clf_votes.reshape(-1, 1)) for clf_votes in votes])
        return onehot_votes

    @staticmethod
    def votes_onehot_to_posteriors(votes):
        committee_size = votes.shape[0]
        return np.apply_along_axis(np.sum, 0, votes) / committee_size

    def __init__(self, base_estimator, target_committee_size, X_train, y_train, x_all_labels, y_all_labels):
        self.base_estimator = base_estimator
        self.target_committee_size = target_committee_size
        self.classifiers = None

        self.X_train = X_train
        self.y_train = y_train
        self.X_all_labels = x_all_labels
        self.y_all_labels = y_all_labels
        return

    def size(self):
        if self.classifiers is not None:
            return len(self.classifiers)
        else:
            raise RuntimeError("committee not trained (no size yet...)")

    @abstractmethod
    def train(self):
        """
        trains the committee
        """
        pass

    def acquire_data(self, X, y):
        self.X_train = np.concatenate([self.X_train, X])
        self.y_train = np.concatenate([self.y_train, y])
        return

    def vote(self, X):
        """
        :param X: data
        :return: votes (classifiers, samples)
        """
        votes = [clf.predict(X) for clf in self.classifiers]
        return np.stack(votes)

    def vote_prob(self, X):
        """
        :param X: data
        :return: votes (classifiers, samples, classes)
        """
        return np.stack([clf.predict_proba(X) for clf in self.classifiers])
