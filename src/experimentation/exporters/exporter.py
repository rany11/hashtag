import os
from abc import ABC, abstractmethod

import pandas


class ExporterKeyError(RuntimeError):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return


class Exporter(ABC):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = {key: list() for key in self._key_set()}
        return

    @abstractmethod
    def get_filename(self):
        """
        :return: the filename relevant to the specific exporter
        """
        pass

    @abstractmethod
    def _key_set(self):
        """
        :return: the set of column names that will be used by this exporter
        """
        pass

    @abstractmethod
    def _metric(self, votes, y_test):
        """
        :param votes: votes of the ensemble for X_test
        :param y_test: true classes (labels)
        :return: the exporter's specific metric for the ensemble's current state
        """
        pass

    def append(self, votes, y_test):
        """
        :param votes: votes of the ensemble for X_test
        :param y_test: true classes (labels)
        appends a measurement of the exporter's specific metric for the ensemble's current state
        """
        for key, val in self._metric(votes, y_test).items():
            self.results[key].append(val)
        return

    def export(self):
        os.makedirs(self.output_dir, exist_ok=True)
        df = pandas.DataFrame.from_dict(self.results)
        df.to_csv(os.path.join(self.output_dir, self.get_filename() + '.csv'))
        return
