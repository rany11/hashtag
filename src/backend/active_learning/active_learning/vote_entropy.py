import numpy as np
from scipy.stats import entropy

from backend.active_learning.active_learning.query_by_committee import QueryByCommittee
from backend.active_learning.committee_utils.committee import Committee


class VoteEntropy(QueryByCommittee):

    def __init__(self):
        super().__init__()
        return

    def strategy_name(self):
        return 'VoteEntropy'

    @staticmethod
    def calc_vote_entropies(votes, committee: Committee):
        posteriors = Committee.votes_onehot_to_posteriors(
            Committee.votes_to_onehot(
                votes,
                committee.y_all_labels
            )
        )
        entropies = np.apply_along_axis(entropy,
                                        1,
                                        posteriors)
        return entropies

    def query_idx(self, committee: Committee, x_unlabeled, nb_queries):
        entropies = VoteEntropy.calc_vote_entropies(committee.vote(x_unlabeled), committee)

        # sort entropies in reverse order (highest entropies are queried), hence the argsort(-entropies)
        return np.argsort(-entropies)[:nb_queries]
