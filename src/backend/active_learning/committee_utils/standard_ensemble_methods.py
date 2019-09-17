from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from committee_utils.committee import Committee


class StandardEnsembleCommittee(Committee, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ = None
        self.bag = self.construct_skl_trainer()
        return

    @abstractmethod
    def construct_skl_trainer(self):
        pass

    def train(self):
        self.bag.fit(np.concatenate((self.X_train, self.X_all_labels)),
                     np.concatenate((self.y_train, self.y_all_labels)))
        self.classifiers = self.bag.estimators_
        self.classes_ = self.bag.classes_
        return

    def vote(self, X):
        # bagging/boosting renames the labels when training, so here we transform the votes back to the original labels
        votes = [self.classes_.take((np.argmax(clf.predict_proba(X), axis=1)), axis=0) for clf in self.classifiers]
        return np.stack(votes)


class CommitteeAdaBoost(StandardEnsembleCommittee):
    def construct_skl_trainer(self):
        return AdaBoostClassifier(self.base_estimator, self.target_committee_size)


class CommitteeBagging(StandardEnsembleCommittee):
    def construct_skl_trainer(self):
        return BaggingClassifier(self.base_estimator, self.target_committee_size)
