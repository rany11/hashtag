from abc import ABC, abstractmethod
from dataclasses import dataclass

from experimentation.configuration.configuration import Configuration
from experimentation.paths import Paths


@dataclass
class ResultsError(ABC):
    exp_name: str
    conf: Configuration
    dataset: str

    @abstractmethod
    def get_path(self):
        pass

    def __lt__(self, other):
        return self.__repr__().__lt__(other.__repr__())


class DatasetResultsMissingRepetitions(ResultsError):
    def get_path(self):
        return Paths.sub_exp_dataset_dir(self.exp_name, self.conf, self.dataset, get_only=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def __repr__(self):
        return f'SubExperimentMissingIterations({self.get_path()})'


class EmptyDatasetResults(DatasetResultsMissingRepetitions):
    def __repr__(self):
        return f'EmptySubExperiment({self.get_path()})'


@dataclass
class RepetitionIncomplete(ResultsError):
    iter_number: int

    def get_path(self):
        return Paths.rep_dir(self.exp_name, self.conf, self.dataset, self.iter_number, get_only=True)

    def __repr__(self):
        return f'IterationIncomplete({self.get_path()})'

