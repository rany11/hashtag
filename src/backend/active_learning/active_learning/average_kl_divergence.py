import numpy as np
from scipy.stats import entropy

from active_learning.query_by_committee import QueryByCommittee
from committee_utils.committee import Committee


class AverageKLDivergence(QueryByCommittee):

    @staticmethod
    def _average_kl_divergence(class_probabilities):
        consensus = np.mean(class_probabilities, axis=0)
        divergence = []
        for y_out in class_probabilities:
            divergence.append(entropy(consensus.T, y_out.T))
        return np.apply_along_axis(np.mean, 0, np.asarray(divergence))

    def query_idx(self, committee: Committee, x_unlabeled, nb_queries):
        probs = committee.vote_prob(x_unlabeled)
        return np.argsort(-AverageKLDivergence._average_kl_divergence(probs))[:nb_queries]

    def strategy_name(self):
        return 'AverageKLDivergence'
