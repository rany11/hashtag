import os
import time
from abc import abstractmethod

from active_learning.average_kl_divergence import AverageKLDivergence
from active_learning.qbcd.combinations.vote_and_diversity_entropy import VoteAndDiversityEntropy
from active_learning.qbcd.diversity_entropy import DiversityEntropy
from active_learning.qbcd.expected_diversity_reduction import MaximumDiversityReduction
from active_learning.qbcd.intra_methods.query_by_y_dist import YDistance
from active_learning.qbcd.intra_methods.y_dist_entropy import YDistEntropy
from active_learning.qbcd.intra_methods.y_dist_max import YDistMax
from active_learning.qbcd.intra_methods.y_dist_mean import YDistMean
from active_learning.qbcd.intra_methods.y_dist_min import YDistMin
from active_learning.qbcd.intra_methods.y_dist_min_margin import YDistMinMargin
from active_learning.query_by_committee import QueryByCommittee
from active_learning.random_sampling import RandomSampling
from active_learning.vote_entropy import VoteEntropy
from committee_utils.decorate_committee import CommitteeDecorate
from committee_utils.diff_clfs_committee import CommitteeDiffClfs
from committee_utils.split_data_committee import CommitteeSplitData
from committee_utils.split_features_committee import CommitteeSplitFeatures
from committee_utils.standard_ensemble_methods import CommitteeBagging, CommitteeAdaBoost
from experimentation.configuration.configuration import Configuration
from experimentation.dataset.dataset import DataSet
from experimentation.dataset.loader import Loader
from experimentation.exporters.implementations import AccuracyPerClass, Accuracy, Diversity, \
    MinClassAccuracy, MSE, CommitteeSize, AccuracyPerClassifier, MinimumSpanningTreeDiversity
from experimentation.logging import Log
from experimentation.paths import Paths, DirAlreadyExists
from experimentation.results_management.utils import do_nothing
import numpy as np


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
            'decorate': CommitteeDecorate,
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

            YDistMin(y_dist_type=np.mean),
            YDistMax(y_dist_type=np.mean),
            YDistMean(y_dist_type=np.mean),
            YDistEntropy(y_dist_type=np.mean),
            YDistMinMargin(y_dist_type=np.mean),

            YDistMin(y_dist_type=np.min),
            YDistMax(y_dist_type=np.min),
            YDistMean(y_dist_type=np.min),
            YDistEntropy(y_dist_type=np.min),
            YDistMinMargin(y_dist_type=np.min),

            YDistMin(y_dist_type=np.max),
            YDistMax(y_dist_type=np.max),
            YDistMean(y_dist_type=np.max),
            YDistEntropy(y_dist_type=np.max),
            YDistMinMargin(y_dist_type=np.max),

            VoteEntropy(),

            DiversityEntropy(use_mst=False),
            DiversityEntropy(use_mst=True),

            MaximumDiversityReduction(use_mst=False),
            MaximumDiversityReduction(use_mst=True),

            VoteAndDiversityEntropy(choose_min_diversity=False),
            VoteAndDiversityEntropy(choose_min_diversity=True),

            # TBD(use_mst=False, div_merge_type=DivMergeType.WEIGHTED, selection_strategy=SelectionStrategy.MIN),
            # TBD(use_mst=False, div_merge_type=DivMergeType.WEIGHTED, selection_strategy=SelectionStrategy.MAX),

            # MinimumDiversityReduction(configuration.vote_size),
            # ByTrainSize(80),
            # SimulatedEstimatedCorrectness(None)  # placeholder
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
