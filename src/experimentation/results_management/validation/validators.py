import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Set, Collection, List

from experimentation.configuration.configuration import Configuration
from experimentation.paths import Paths
from experimentation.results_management.validation.errors import RepetitionIncomplete, EmptyDatasetResults, \
    DatasetResultsMissingRepetitions
from main import manual_configuration


class Validator(ABC):
    @abstractmethod
    def get_info(self):
        pass

    def propagate_info(self, propagated_info: Dict):
        propagated_info[self.__class__].update(self.get_info())
        return

    def check_error(self, propagated_info: Dict):
        diff = propagated_info[self.__class__].difference(self.get_info())
        return len(propagated_info[self.__class__].difference(self.get_info())) > 0

    @abstractmethod
    def create_error(self, propagated_info):
        pass

    def propagate_errors(self, propagated_info, propagated_errors):
        e = self.create_error(propagated_info)
        if e is not None:
            propagated_errors[self.__class__].append(e)
        return


class StrategyResults(Validator):
    strategy_name: str
    filenames: Collection[str]

    def create_error(self, propagated_info):
        pass

    def get_info(self):
        return self.filenames

    def __init__(self, exp_name, conf, dataset, rep_i, strategy_name):
        self.strategy_name = strategy_name
        self.filenames = os.listdir(
            os.path.join(Paths.rep_dir(exp_name, conf, dataset, rep_i, get_only=True), strategy_name)
        )
        return


class RepetitionResults(Validator):
    strategies: Collection[StrategyResults]
    rep_i: int

    def create_error(self, propagated_info):
        error = RepetitionIncomplete(*self.details)

        if self.check_error(propagated_info):
            return error

        # # the following error checkiing is problematic
        # # because in some cases 'accuracy_per_classifier.csv' is present,
        # # which makes all other sub-experiments seem problematic
        # else:
        #     for s in self.strategies:
        #         if s.check_error(propagated_info):
        #             return error
        return None

    def get_info(self):
        return [s.strategy_name for s in self.strategies]

    def propagate_info(self, propagated_info: Dict):
        super(RepetitionResults, self).propagate_info(propagated_info)
        for s in self.strategies:
            s.propagate_info(propagated_info)
        return

    def __init__(self, exp_name, conf, dataset, rep_dir):
        self.rep_i = Paths.rep_idx_from_dir_name(rep_dir)
        self.strategies = [
            StrategyResults(exp_name, conf, dataset, self.rep_i, strategy_name)
            for strategy_name in Paths.list_only_dirs(Paths.rep_dir(exp_name, conf, dataset, self.rep_i, get_only=True))
        ]
        self.details = (exp_name, conf, dataset, self.rep_i)
        return


class DatasetResults(Validator):
    rep_results: Collection[RepetitionResults]
    rep_ids: List[int]

    def create_error(self, propagated_info):
        if len(self.rep_ids) == 0:
            return EmptyDatasetResults(*self.details)
        elif self.check_error(propagated_info):
            return DatasetResultsMissingRepetitions(*self.details)
        else:
            return None

    def propagate_errors(self, propagated_info, propagated_errors):
        super().propagate_errors(propagated_info, propagated_errors)
        for rr in self.rep_results:
            rr.propagate_errors(propagated_info, propagated_errors)
        return

    def get_info(self):
        return self.rep_ids

    def propagate_info(self, propagated_info: Dict):
        if len(self.rep_results) > 0:
            propagated_info[self.__class__] = max(propagated_info.get(self.__class__, -1), *self.get_info())
        for rr in self.rep_results:
            rr.propagate_info(propagated_info)
        return

    def check_error(self, propagated_info: Dict):
        return max(self.get_info()) < propagated_info[self.__class__]

    def __init__(self, exp_name, conf, dataset):
        self.rep_results = [
            RepetitionResults(exp_name, conf, dataset, rep_dir)
            for rep_dir in Paths.list_only_dirs(
                Paths.sub_exp_dataset_dir(exp_name, conf, dataset, get_only=True)
            )
            if rep_dir != Paths.AVERAGE_RESULTS_DIR_NAME
        ]

        self.rep_ids = [rep.rep_i for rep in self.rep_results]
        self.details = (exp_name, conf, dataset)
        return

    def nb_repetitions(self):
        return len(self.rep_results)


class SubExperimentResults(Validator):
    datasets_results: List[DatasetResults]
    dataset_names: List[str]

    def create_error(self, propagated_info):
        pass

    def propagate_errors(self, propagated_info, propagated_errors):
        for dr in self.datasets_results:
            dr.propagate_errors(propagated_info, propagated_errors)
        return

    def get_info(self):
        return self.dataset_names

    def propagate_info(self, propagated_info: Dict):
        for dr in self.datasets_results:
            dr.propagate_info(propagated_info)
        return

    def __init__(self, exp_name, sub_exp):
        sub_exp_dir = Paths.sub_exp_dir(exp_name, sub_exp, get_only=True)

        self.conf = Configuration.from_json_path(os.path.join(sub_exp_dir, Paths.CONFIG_FILENAME))
        self.dataset_names = [f for f in Paths.list_only_dirs(sub_exp_dir) if f != Paths.MERGED_RESULTS_DIR_NAME]
        self.datasets_results = [DatasetResults(exp_name, self.conf, dataset) for dataset in self.dataset_names]
        return

    def get_nb_repetitions(self):
        return sum([dr.nb_repetitions() for dr in self.datasets_results])


class ExperimentResults:
    sub_exp_results: List[SubExperimentResults]

    def __init__(self, exp_name):
        self.sub_exp_results = list()
        for sub_exp in Paths.list_only_dirs(Paths.exp_dir(exp_name, get_only=True)):
            self.sub_exp_results.append(SubExperimentResults(exp_name, sub_exp))
        return

    def find_errors(self):
        d_info = defaultdict(set)
        d_errors = defaultdict(list)

        for r in self.sub_exp_results:
            r.propagate_info(d_info)
        for r in self.sub_exp_results:
            r.propagate_errors(d_info, d_errors)
        return sum(d_errors.values(), [])

    def get_statisitcs(self):
        return len(self.sub_exp_results), sum([ser.get_nb_repetitions() for ser in self.sub_exp_results])
