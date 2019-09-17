import numpy as np

from backend.active_learning.committee_utils.committee import Committee
from backend.active_learning.experimentation.exporters.exporter import Exporter, ExporterKeyError


def accuracy(y_predict, y_true):
    total_correct = (y_predict == y_true).astype(np.int).sum()
    total_size = y_true.shape[0]
    return total_correct / total_size


def mean_squared_error(y_predict_onehot, y_true_onehot):
    y_predict_prob = Committee.votes_onehot_to_posteriors(y_predict_onehot)
    return ((y_predict_prob - y_true_onehot) ** 2).mean()


class AccuracyPerClassifier(Exporter):
    def __init__(self, ensemble_size, *args, **kwargs):
        self.max_idx = ensemble_size - 1
        super().__init__(*args, **kwargs)
        return

    def __key(self, i):
        if i > self.max_idx:
            raise ExporterKeyError("AccuracyPerClassifier: {} > max_idx ({})".format(i, self.max_idx))
        return 'clf_{}_accuracy'.format(i)

    def _key_set(self):
        return [self.__key(i) for i in range(self.max_idx + 1)]  # 0.. max_idx

    def _metric(self, votes, y_test):
        acc_per_clf = dict()
        for i, clf_votes in enumerate(votes):
            acc_per_clf[self.__key(i)] = accuracy(clf_votes, y_test)

        return acc_per_clf

    def get_filename(self):
        return 'accuracy_per_classifier'


class AccuracyPerClass(Exporter):
    def __init__(self, all_labels, *args, **kwargs):
        self.all_labels = all_labels
        super().__init__(*args, **kwargs)
        return

    def __key(self, label):
        return 'class_{}_accuracy'.format(label)

    def _key_set(self):
        return [self.__key(label) for label in self.all_labels]

    def _metric(self, votes, y_test):
        acc_per_class = dict()
        maj_votes = Committee.majority(votes)
        for label in self.all_labels:
            l_idx = (y_test == label)
            acc = accuracy(maj_votes[l_idx], y_test[l_idx])
            acc_per_class[self.__key(label)] = acc

        return acc_per_class

    def get_filename(self):
        return 'accuracy_per_class'


class MinClassAccuracy(AccuracyPerClass):
    def _key(self):
        return 'min_class_accuracy'

    def _key_set(self):
        return [self._key()]

    def _metric(self, votes, y_test):
        return {self._key(): min(super()._metric(votes, y_test).values())}

    def get_filename(self):
        return 'min_class_accuracy'


class Accuracy(Exporter):
    __ACCURACY_KEY = 'accuracy'

    def _key_set(self):
        return [Accuracy.__ACCURACY_KEY]

    def _metric(self, votes, y_test):
        return {Accuracy.__ACCURACY_KEY: accuracy(Committee.majority(votes), y_test)}

    def get_filename(self):
        return 'accuracy'


class MSE(Exporter):
    __KEY = 'mse'

    def __init__(self, output_dir, all_labels):
        super().__init__(output_dir)
        self.__all_labels = all_labels
        return

    def _key_set(self):
        return [MSE.__KEY]

    def _metric(self, votes, y_test):
        return {
            MSE.__KEY:
                mean_squared_error(
                    Committee.votes_to_onehot(votes, self.__all_labels),
                    Committee.votes_to_onehot(np.array([y_test]), self.__all_labels)[0]
                )
        }

    def get_filename(self):
        return 'mean_squared_error'


class Diversity(Exporter):
    __DIVERSITY_KEY = 'diversity'

    def _key_set(self):
        return [Diversity.__DIVERSITY_KEY]

    def _metric(self, votes, y_test):
        nb_voters = votes.shape[0]
        dis_mat = distance_matrix(nb_voters, votes)
        voters = np.arange(nb_voters)
        div = diversity(dis_mat, voters)
        return {Diversity.__DIVERSITY_KEY: div}

    def get_filename(self):
        return 'diversity'


class MinimumSpanningTreeDiversity(Exporter):
    __MST_DIVERSITY_KEY = 'mst-diversity'

    def _key_set(self):
        return [MinimumSpanningTreeDiversity.__MST_DIVERSITY_KEY]

    def _metric(self, votes, y_test):
        nb_voters = votes.shape[0]
        dis_mat = distance_matrix(nb_voters, votes)
        div = mst_diversity(dis_mat, np.arange(nb_voters))
        return {MinimumSpanningTreeDiversity.__MST_DIVERSITY_KEY: div}

    def get_filename(self):
        return 'mst_diversity'


class CommitteeSize(Exporter):
    def get_filename(self):
        return 'committee_size'

    def _key_set(self):
        return ['committee_size']

    def _metric(self, votes, y_test):
        return {'committee_size': votes.shape[0]}
