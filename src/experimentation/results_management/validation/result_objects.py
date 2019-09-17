import os
from abc import ABC, abstractmethod
from collections import defaultdict
from os.path import join
from sys import stderr
from typing import Dict, Set, Collection, List

from experimentation.configuration.configuration import Configuration
from experimentation.paths import Paths
from experimentation.results_management.utils import sets_differ, delete_path
from experimentation.results_management.validation.errors import RepetitionIncomplete, EmptyDatasetResults, \
    DatasetResultsMissingRepetitions
from experimentation.utils import t_confidence_interval_thunk
import pandas as pd


class ColumnNames:
    DATASET = 'dataset'
    METRIC = 'metric'
    AGGREGATION_TYPE = 'agg'
    STRATEGY = 'strategy'
    ENSEMBLE_METHOD = 'training_method'
    QUERY_NUMBER = 'query'


class StrategyResults:
    def __init__(self, exp_name, conf, dataset, rep_i, strategy_name):
        self.strategy_name = strategy_name
        self.dir = join(Paths.rep_dir(exp_name, conf, dataset, rep_i, get_only=True), strategy_name)
        return

    def get_exports(self):
        return os.listdir(self.dir)

    def merge(self):
        dfs = [pd.read_csv(join(self.dir, filename), index_col=0) for filename in self.get_exports()]
        df = pd.concat(dfs, axis='columns')
        df.rename_axis(ColumnNames.QUERY_NUMBER, inplace=True)
        return df


class RepetitionResults:
    def __init__(self, exp_name, conf, dataset, rep_dir):
        rep_i = Paths.rep_idx_from_dir_name(rep_dir)
        self.info = (exp_name, conf, dataset, rep_i)
        self.strategy_results = [
            StrategyResults(exp_name, conf, dataset, rep_i, strategy_name)
            for strategy_name in Paths.list_only_dirs(self.path())
        ]
        return

    def rep_i(self):
        return self.info[-1]

    def path(self):
        return Paths.rep_dir(*self.info, get_only=True)

    def get_strategy_names(self):
        return [sr.strategy_name for sr in self.strategy_results]

    def check_errors(self, expected_strategies, expected_exports):
        error = RepetitionIncomplete(*self.info)

        if sets_differ(expected_strategies, self.get_strategy_names()):
            return error

        for sr in self.strategy_results:
            if sets_differ(expected_exports, sr.get_exports()):
                return error

        return None

    def merge(self):
        df = pd.concat(
            [sr.merge() for sr in self.strategy_results],
            keys=[sr.strategy_name for sr in self.strategy_results],
            names=[ColumnNames.STRATEGY, ColumnNames.METRIC],
            axis='columns'
        )
        return df


class DatasetResults:
    def __init__(self, exp_name, conf, dataset):
        self.sub_exp_info = (exp_name, conf)
        self.dataset_name = dataset

        self.rep_results = [RepetitionResults(exp_name, conf, dataset, rep_dir)
                            for rep_dir in Paths.list_only_dirs(self.path())
                            if rep_dir != Paths.AVERAGE_RESULTS_DIR_NAME]
        return

    def path(self):
        return Paths.sub_exp_dataset_dir(*self.sub_exp_info, self.dataset_name, get_only=True)

    def get_rep_ids(self):
        return [rr.rep_i() for rr in self.rep_results]

    def merge(self):
        df = pd.concat([rr.merge() for rr in self.rep_results],
                       axis='columns',
                       keys=self.get_rep_ids(),
                       names=['rep_i'])

        agg_names = ['max', 'min', 'mean',
                     # 't confidence interval (0.95)'
                     ]
        agg_funcs = [pd.DataFrame.max, pd.DataFrame.min, pd.DataFrame.mean,
                     # t_confidence_interval_thunk(0.95)
                     ]
        aggregated_dfs = [df.apply(func,
                                   axis='columns',
                                   level=[ColumnNames.STRATEGY, ColumnNames.METRIC])
                          for func in agg_funcs]

        joined = pd.concat(aggregated_dfs,
                           keys=agg_names,
                           names=[ColumnNames.AGGREGATION_TYPE],
                           axis='columns')
        return joined

    def find_errors(self, expected_strategies, expected_exports):
        errors = list()
        for rr in self.rep_results:
            e = rr.check_errors(expected_strategies, expected_exports)
            if e is not None:
                errors.append(e)
        return errors

    def clean(self, expected_strategies, expected_exports):
        for rr in self.rep_results:
            if rr.check_errors(expected_strategies, expected_exports) is not None:
                delete_path(rr.path())
                self.rep_results.remove(rr)
        return


class SubExperimentResults:
    def __init__(self, exp_name, sub_exp):
        self.dir = Paths.sub_exp_dir(exp_name, sub_exp, get_only=True)
        self.conf = Configuration.from_json_path(os.path.join(self.dir, Paths.CONFIG_FILENAME))
        self.dataset_names = [f for f in Paths.list_only_dirs(self.dir) if f != Paths.MERGED_RESULTS_DIR_NAME]
        self.datasets_results = [DatasetResults(exp_name, self.conf, dataset) for dataset in self.dataset_names]
        return

    def path(self):
        return self.dir

    def merge(self):
        df = pd.concat(
            [dr.merge() for dr in self.datasets_results],
            keys=[dr.dataset_name for dr in self.datasets_results],
            names=[ColumnNames.DATASET],
            axis='columns'
        )
        return df

    def gather_info(self):
        highest_rep_expected = -1
        expected_strategies = set()
        expected_exports = set()
        for dr in self.datasets_results:
            highest_rep_expected = max(dr.get_rep_ids())

            for rr in dr.rep_results:
                expected_strategies.update(rr.get_strategy_names())

                for sr in rr.strategy_results:
                    expected_exports.update(sr.get_exports())

        return highest_rep_expected, expected_strategies, expected_exports

    def find_errors(self, highest_rep_expected: int, expected_strategies: Set, expected_exports: Set):
        errors = list()
        for dr in self.datasets_results:
            iteration_errors = dr.find_errors(expected_strategies, expected_exports)
            errors.extend(iteration_errors)

            nb_complete_reps = len(dr.get_rep_ids()) - len(iteration_errors)
            if nb_complete_reps == 0:
                errors.append(EmptyDatasetResults(*dr.sub_exp_info, dr.dataset_name))
            elif nb_complete_reps < highest_rep_expected + 1:
                errors.append(DatasetResultsMissingRepetitions(*dr.sub_exp_info, dr.dataset_name))
        return errors

    def clean(self, highest_rep_expected: int, expected_strategies: Set, expected_exports: Set):
        for dr in self.datasets_results:
            dr.clean(expected_strategies, expected_exports)
            if len(dr.get_rep_ids()) == 0:
                delete_path(dr.path())
                self.datasets_results.remove(dr)
        return


class ExperimentResults:
    def __init__(self, exp_name):
        self.dir = Paths.exp_dir(exp_name, get_only=True)
        self.sub_exp_results = [SubExperimentResults(exp_name, sub_exp) for sub_exp in Paths.list_only_dirs(self.dir)]
        return

    @staticmethod
    def config_to_concat_keys(conf, keys, parames_to_ignore, parames_to_string):
        d = conf.__dict__.copy()
        if set(d.keys()).difference(parames_to_ignore) != keys:
            raise RuntimeError("configuration keys don't match")

        for p in parames_to_string:
            d[p] = str(d[p])

        return tuple(d[k] for k in keys)

    def get_statistics(self):
        nb_sub_results = len(self.sub_exp_results)
        nb_reps = sum([
            sum([len(dr.rep_results) for dr in ser.datasets_results])
            for ser in self.sub_exp_results]
        )
        return nb_sub_results, nb_reps

    def merge(self):
        configs = [ser.conf for ser in self.sub_exp_results]
        params_to_ignore = ['datasets', 'query_measure_points']
        params_to_string = ['base_estimator']
        conf_keys = set(configs[0].__dict__.keys()).difference(params_to_ignore)

        ser_dfs = list()
        for i, ser in enumerate(self.sub_exp_results, 1):
            try:
                ser_dfs.append(ser.merge())
                print(f'{ser.path()} ({i}/{len(self.sub_exp_results)})')
            except:
                print(f'some problem with {ser.path()}', file=stderr)
        df = pd.concat(
            ser_dfs,
            keys=[ExperimentResults.config_to_concat_keys(conf, conf_keys, params_to_ignore, params_to_string)
                  for conf in configs],
            names=list(conf_keys),
            axis='columns'
        )
        return df

    def pivot(self):
        df = self.merge()
        return df.stack(level=list(set(df.columns.names).difference([ColumnNames.METRIC])))

    def find_errors(self):
        """
        detects empty sub experiments (per dataset), incomplete sub experiments (per dataset, compared to the other
        datasets in the same configuration), and incomplete repetitions (compared to other repetitions in the same
        sub experiments)

        :return: list of errors
        """
        return sum([ser.find_errors(*ser.gather_info()) for ser in self.sub_exp_results], [])

    def find_errors_between_subs(self):
        """
        same as 'find_errors', only expects the same number of repetitions for sub-experiments
        :return: list of errors
        """
        highest_reps, strategies_all, exports_all = zip(*[ser.gather_info() for ser in self.sub_exp_results])
        highest_rep_i = max(highest_reps)
        errors = sum([ser.find_errors(highest_rep_i, strategies, exports)
                      for ser, strategies, exports in zip(self.sub_exp_results, strategies_all, exports_all)], [])
        return errors

    def clean(self):
        for ser in self.sub_exp_results:
            ser.clean(*ser.gather_info())
            if len(ser.datasets_results) == 0:
                delete_path(ser.path())
                self.sub_exp_results.remove(ser)
        return
