from abc import ABC, abstractmethod

from backend.active_learning.committee_utils.committee import Committee


class QueryByCommittee(ABC):
    """
    (abstract class)
    represents a single QBC strategy
    """

    def __init__(self):
        return

    def __str__(self):
        return self.strategy_name()

    def __repr__(self):
        return self.__str__()

    @abstractmethod
    def query_idx(self, committee: Committee, x_unlabeled, nb_queries):
        """
        :param committee: trained ensemble
        :param x_unlabeled: unlabeled dataset to query samples from
        :param nb_queries: number of samples to query for
        :return: indexes of the samples queried by the QBC method
        """
        pass

    @abstractmethod
    def strategy_name(self):
        pass
