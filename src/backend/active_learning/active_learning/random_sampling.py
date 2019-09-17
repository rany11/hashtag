import numpy as np

from backend.active_learning.active_learning.query_by_committee import QueryByCommittee
from backend.active_learning.committee_utils.committee import Committee


class RandomSampling(QueryByCommittee):
    def strategy_name(self):
        return 'RandomSampling'

    def __init__(self):
        super().__init__()
        return

    def query_idx(self, committee: Committee, x_unlabeled, nb_queries):
        return np.random.choice(x_unlabeled.shape[0], size=nb_queries, replace=False)
