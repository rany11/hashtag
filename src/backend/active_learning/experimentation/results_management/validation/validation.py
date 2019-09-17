import os

from experimentation.paths import Paths
from experimentation.results_management.validation.errors import RepetitionIncomplete, DatasetResultsMissingRepetitions


def validate_experiment(exp_name):
    found_errors = []
    iters_counter = 0
    configs = Paths.get_exp_configs(exp_name)
    for conf in configs:
        sub_exp_errors, sub_exp_iters = validate_sub_exp(exp_name, conf)
        found_errors.extend(sub_exp_errors)
        iters_counter += sub_exp_iters

    return sorted(found_errors), len(configs), iters_counter


def validate_sub_exp(exp_name, conf):
    found_errors = []
    iters_counter = 0
    datasets = [f for f in Paths.list_only_dirs(Paths.sub_exp_dir_by_conf(exp_name, conf))
                if f != Paths.MERGED_RESULTS_DIR_NAME]
    for dataset in datasets:
        dataset_errors, nb_dataset_iters = validate_sub_exp_dataset(exp_name, conf, dataset)
        found_errors.extend(dataset_errors)
        iters_counter += nb_dataset_iters

    return found_errors, iters_counter


def validate_sub_exp_dataset(exp_name, conf, dataset):
    found_errors = []

    existing_iters = []
    iter_dirs = []
    for iter_number in range(conf.nb_repeats):
        dir = Paths.rep_dir(exp_name, conf, dataset, iter_number, get_only=True)
        if os.path.exists(dir):
            existing_iters.append(iter_number)
            iter_dirs.append(dir)

    strategie_counts = []
    export_counts = []
    max_export_count = 0
    for dir in iter_dirs:
        strategies = Paths.list_only_dirs(dir)

        nb_exports = [len(os.listdir(os.path.join(dir, strategy)))
                      for strategy in strategies]

        strategie_counts.append(len(strategies))
        max_export_count = max(nb_exports + [max_export_count])
        export_counts.append(min(nb_exports) if len(nb_exports) > 0 else 0)

    if len(existing_iters) < conf.nb_repeats:
        found_errors.append(DatasetResultsMissingRepetitions(exp_name, conf, dataset))

    for iter_number, nb_strs, nb_exps in zip(existing_iters, strategie_counts, export_counts):
        if nb_strs < max(strategie_counts) or nb_exps < max_export_count:
            found_errors.append(RepetitionIncomplete(exp_name, conf, dataset, iter_number))
    return found_errors, len(existing_iters)


def get_rep_strategies(exp_name, conf, dataset, repetition):
    return Paths.list_only_dirs(Paths.rep_dir(exp_name, conf, dataset, repetition))


def get_strategy_exports(exp_name, conf, dataset, repetition, strategy):
    return os.listdir(os.path.join(Paths.rep_dir(exp_name, conf, dataset, repetition), strategy))
