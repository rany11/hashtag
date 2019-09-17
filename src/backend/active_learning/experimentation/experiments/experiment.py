import os
import time
from abc import abstractmethod

import numpy as np

from src.backend.active_learning.active_learning.average_kl_divergence import AverageKLDivergence
from src.backend.active_learning.active_learning.query_by_committee import QueryByCommittee
from src.backend.active_learning.active_learning.random_sampling import RandomSampling
from src.backend.active_learning.active_learning.vote_entropy import VoteEntropy
from src.backend.active_learning.committee_utils.diff_clfs_committee import CommitteeDiffClfs
from src.backend.active_learning.committee_utils.split_data_committee import CommitteeSplitData
from src.backend.active_learning.committee_utils.split_features_committee import CommitteeSplitFeatures
from src.backend.active_learning.committee_utils.standard_ensemble_methods import CommitteeBagging, CommitteeAdaBoost
from src.backend.active_learning.experimentation.configuration.configuration import Configuration
from src.backend.active_learning.experimentation.dataset.dataset import DataSet
from src.backend.active_learning.experimentation.dataset.loader import Loader
from src.backend.active_learning.experimentation.exporters.implementations import Accuracy, AccuracyPerClass, \
    MinClassAccuracy, MSE, CommitteeSize, Diversity, MinimumSpanningTreeDiversity, AccuracyPerClassifier
from src.backend.active_learning.experimentation.logging import Log
from src.backend.active_learning.experimentation.paths import Paths, DirAlreadyExists
from src.backend.active_learning.experimentation.results_management.utils import do_nothing


class Experiment:
    __RESULTS_DIR = r'experiments_results'

    def __init__(self, id):
        self.id = id
        return

    @staticmethod
    def time_as_string(seconds):
        seconds = int(seconds)
        fmt_str = '{hours}:{minutes}:{seconds}'
        return fmt_str.format(hours=seconds//3600, minutes=(seconds % 3600) // 60, seconds=seconds % 60)

    @staticmethod
    def write_configuration(experiment_dir, configuration):
        with open(Paths.sub_exp_conf_path(experiment_dir, configuration), 'w') as f:
            f.write(configuration.to_json_text())
        return

    @staticmethod
    def committee_constructor_dispatch(training_method):
        def unknown_meta_learner(*args, **kwargs):
            raise RuntimeError('unknown training method "{}"'.format(training_method))

        constructor = {
            'split-data': CommitteeSplitData,
            'split-features': CommitteeSplitFeatures,
            'bagging': CommitteeBagging,
            'adaboost': CommitteeAdaBoost,
            'diff-clfs': CommitteeDiffClfs,
        }.get(training_method, unknown_meta_learner)
        return constructor

    def run(self, config: Configuration, verbose):
        """
        :param config: the configuration of the experiment
        runs the experiment
        """
        Experiment.write_configuration(self.id, config)  # write configuration file
        if not verbose:
            might_print = do_nothing
        else:
            might_print = print

        strategies = [
            RandomSampling(),
            VoteEntropy(),
            AverageKLDivergence()
        ]
        if config.training_method != 'diff-clfs':
            strategies.append(AverageKLDivergence())

        for dataset_name in config.datasets:
            data, target = Loader.load_dataset(dataset_name)  # load the dataset

            total_running_time = 0
            total_busy_repetitions = 0
            for repetition_n in range(config.nb_repeats):
                might_print('{} (repetition {}/{})...'.format(dataset_name, repetition_n + 1, config.nb_repeats))

                # the directory for the exporters to use
                try:
                    current_rep_out_dir = Paths.rep_dir(self.id,
                                                        config,
                                                        dataset_name,
                                                        repetition_n)
                except DirAlreadyExists:
                    might_print('repetition already exists...')
                    might_print()
                    continue

                log = Log(current_rep_out_dir)

                # the DataSet object shuffles the dataset and separates the data into train, test and so on...
                dataset = DataSet(data,
                                  target,
                                  config)
                log.log(dataset.data_summary, quite=True)

                rep_total_time = 0
                for current_strategy in strategies:
                    log.log('current strategy: {}'.format(current_strategy), quite=not verbose)
                    elapsed_seconds = self.conduct_single_strategy_experiment(config,
                                                                              current_strategy,
                                                                              dataset,
                                                                              os.path.join(current_rep_out_dir,
                                                                                           current_strategy.strategy_name()))
                    rep_total_time += elapsed_seconds
                    log.log(f'done. elapsed time: {Experiment.time_as_string(elapsed_seconds)}', quite=not verbose)

                total_busy_repetitions += 1
                total_running_time += rep_total_time
                might_print('total time for repetition {} is {}'.format(repetition_n + 1,
                                                                  Experiment.time_as_string(rep_total_time)))
                might_print('estimated time left: {}'.format(Experiment.time_as_string(
                    (config.nb_repeats - repetition_n - 1) * (total_running_time / total_busy_repetitions)
                )))
                might_print()
        return

    def conduct_single_strategy_experiment(self, configuration, strategy, dataset, output_dir):
        # initialize the exporters
        exporters = [
            Accuracy(output_dir),
            AccuracyPerClass(dataset.all_labels, output_dir),
            MinClassAccuracy(dataset.all_labels, output_dir),
            MSE(output_dir, dataset.all_labels),
            CommitteeSize(output_dir),
            Diversity(output_dir),
            MinimumSpanningTreeDiversity(output_dir),
        ]
        if configuration.training_method != 'adaboost':
            exporters.append(AccuracyPerClassifier(configuration.target_committee_size, output_dir))

        # main loop:
        #   1. execute active learning simulation iteration
        #   2. measure metrics
        start_time = time.time()
        for committee in self.simulate_active_learning_scenario(configuration,
                                                                strategy,
                                                                dataset):
            for exporter in exporters:
                exporter.append(committee.vote(dataset.X_test), dataset.y_test)
        finish_time = time.time()

        # export results (write to disk)
        for exporter in exporters:
            exporter.export()

        return finish_time - start_time

    @abstractmethod
    def simulate_active_learning_scenario(self,
                                          configuration: Configuration,
                                          strategy: QueryByCommittee,
                                          dataset):
        """
        yields a trained committee object after each train-query-oracle iteration

        :param configuration: experiment configuration
        :param strategy: active learning strategy
        :param dataset: DataSet object
        :return: trained committee iterable
        """
        pass
