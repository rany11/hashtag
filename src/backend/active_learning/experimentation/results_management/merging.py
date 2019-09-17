import itertools
import os
from multiprocessing.pool import Pool
from sys import stderr

import pandas as pd
import numpy as np

from experimentation.configuration.configuration import Configuration
from experimentation.paths import Paths
from experimentation.results_management.utils import do_nothing
from experimentation.utils import t_confidence_interval_thunk
from main.setup import MAX_JOBS


class ColumnNames:
    DATASET = 'dataset'
    METRIC = 'metric'
    AGGREGATION_TYPE = 'agg'
    STRATEGY = 'strategy'
    ENSEMBLE_METHOD = 'training_method'
    QUERY_RANGE = 'query_range'


def merge_sub_exp_dataset_results(exp_name, config: Configuration, strategy_names, dataset_name,
                                  flexible=False, verbose=True):
    if not verbose:
        print = do_nothing

    print(f'- merging results for dataset {dataset_name}:')
    # TODO: retrieve dynamically
    exported_filenames = [
        # 'accuracy_per_classifier',
        'accuracy_per_class',
        'min_class_accuracy',
        'accuracy',
        'mean_squared_error',
        'diversity',
        'mst_diversity',
        'committee_size'
    ]
    exported_filenames = [fn + '.csv' for fn in exported_filenames]

    average_results_dir_fmt = os.path.join(
        Paths.avg_res_dir(exp_name, config, dataset_name),
        '{strategy_name}'
    )

    for strategy_name in strategy_names:
        strategy_output_dir = average_results_dir_fmt.format(strategy_name=strategy_name)
        print(f'- * averaging iterations of {strategy_name}')
        average_strategy_results(strategy_name, exp_name, config, dataset_name,
                                 exported_filenames, config.nb_repeats, strategy_output_dir,
                                 flexible=flexible)

    print(f"- * creating '{Paths.MERGED_RESULTS_DIR_NAME}' dir...")
    merge_strategies(
        exp_name,
        config,
        dataset_name,
        strategy_names,
        exported_filenames,
        average_results_dir_fmt
    )
    print('- done')
    return


def merge_strategies(exp_name, config, dataset_name, strategy_names, exported_filenames, average_results_dir_fmt):
    merged_csv_path_format = os.path.join(
        Paths.dataset_merged_results_dir(exp_name, config, dataset_name),
        '{filename}'
    )

    for filename in exported_filenames:
        dfs = [
            pd.read_csv(
                os.path.join(average_results_dir_fmt.format(strategy_name=strategy_name), filename),
                index_col=0,
                header=[0, 1]
            )
            for strategy_name in strategy_names
        ]
        merged = pd.concat(dfs, axis=1, keys=strategy_names, names=[ColumnNames.STRATEGY])
        merged.to_csv(merged_csv_path_format.format(filename=filename))
    return


def average_strategy_results(strategy,
                             exp_name,
                             conf,
                             dataset,
                             exported_filenames,
                             nb_repetitions,
                             strategy_output_dir,
                             flexible=False):

    os.makedirs(strategy_output_dir, exist_ok=True)

    for filename in exported_filenames:
        csv_paths = [
            os.path.join(Paths.rep_dir(exp_name, conf, dataset, repetition_n, get_only=True), strategy, filename)
            for repetition_n in range(nb_repetitions)
        ]
        csvs = [
            pd.read_csv(
                path,
                index_col=0
            )
            for path in csv_paths
            if os.path.exists(path) or not flexible
        ]
        all_iters_df = pd.concat(csvs, axis=1)

        mean = pd.DataFrame({c: all_iters_df[c].mean(axis='columns') for c in all_iters_df.columns})
        min = pd.DataFrame({c: all_iters_df[c].min(axis='columns') for c in all_iters_df.columns})
        max = pd.DataFrame({c: all_iters_df[c].max(axis='columns') for c in all_iters_df.columns})
        conf_int = pd.DataFrame({
            c: all_iters_df[c].apply(t_confidence_interval_thunk(0.95), axis='columns')
            for c in all_iters_df.columns
        })

        joined = pd.concat([min, max, mean, conf_int],
                           axis=1,
                           keys=['min', 'max', 'mean', 't conf margin'],
                           names=[ColumnNames.AGGREGATION_TYPE, ColumnNames.METRIC])

        # joined = joined.swaplevel(0, 1, axis=1)
        joined.to_csv(os.path.join(strategy_output_dir, filename))
    return


def star_get_merge_all_datasets(args):
    try:
        return get_merged_all_datasets(*args)
    except:
        return None


def merge_configs(experiment_name):
    configs = Paths.get_exp_configs(experiment_name)
    print(f'merging configs of experiment {experiment_name}:')
    print()
    all_configs_dfs = dict()

    with Pool(min(MAX_JOBS, len(configs))) as pool:
        pool_res = pool.imap(
            star_get_merge_all_datasets,
            [(experiment_name, conf, False) for conf in configs],
            chunksize=1
        )
        for i, conf, config_result_dfs in zip(itertools.count(), configs, pool_res):
            if config_result_dfs is None:
                print(f'some problem with experiment {Paths.config_as_dir_name(conf)}', file=stderr)
                continue

            print(f'merging configuration {Paths.config_as_dir_name(conf)} ({i+1}/{len(configs)})')

            config_result_dfs = {
                filename: df.assign(**{ColumnNames.QUERY_RANGE: df.index}).melt(id_vars=ColumnNames.QUERY_RANGE)
                for filename, df in config_result_dfs.items()
            }

            parameters_to_ignore = ['datasets', 'query_measure_points']
            for filename, df in config_result_dfs.items():
                d = conf.__dict__.copy()
                d['base_estimator'] = str(d['base_estimator'])
                for p in parameters_to_ignore:
                    del d[p]

                d = {k: np.repeat(v, len(df))
                     for k, v in d.items()}

                config_result_dfs[filename] = df.assign(**d)

            for filename, df in config_result_dfs.items():
                all_configs_dfs[filename] = all_configs_dfs.get(filename, list())
                all_configs_dfs[filename].append(df)

    all_configs_dfs = {filename: pd.concat(dfs)
                       for filename, dfs in all_configs_dfs.items()}

    pd.concat(all_configs_dfs.values()).to_csv(Paths.get_pivot_file_path(experiment_name), index=False)
    print('done.')
    return


def get_merged_all_datasets(exp_name, conf, verbose=True):
    if not verbose:
        print = do_nothing

    datasets = [f for f in Paths.list_only_dirs(Paths.sub_exp_dir_by_conf(exp_name, conf))
                if f != Paths.MERGED_RESULTS_DIR_NAME]
    strategy_names = Paths.list_only_dirs(
        Paths.rep_dir(exp_name, conf, datasets[0], 0, get_only=True)
    )
    for i, dataset in enumerate(datasets):
        print(f'dataset {i+1}/{len(datasets)}')
        merge_sub_exp_dataset_results(exp_name, conf, strategy_names, dataset, flexible=True, verbose=verbose)

    merged_res_dir = Paths.merged_results_dir(exp_name, conf)
    dataset_names = os.listdir(merged_res_dir)
    if len(dataset_names) < 1:
        return None

    filenames = os.listdir(os.path.join(merged_res_dir, dataset_names[0]))
    all_dfs = {fn: list() for fn in filenames}
    for fn in filenames:
        for dn in dataset_names:
            df = pd.read_csv(
                os.path.join(merged_res_dir, dn, fn),
                index_col=0,
                header=[0, 1, 2]
            )
            df = df.apply(sub_means_gen(5))
            print(df)
            all_dfs[fn].append(df)

    merged_dfs = {
        filename: pd.concat(dfs, axis=1, keys=dataset_names, names=[ColumnNames.DATASET], sort=False)
        for filename, dfs in all_dfs.items()
    }
    print(merged_dfs)
    return merged_dfs


def sub_means_gen(subset_size):
    def sub_means(x):
        x = np.array(x)
        tail_length = x.shape[0] % subset_size
        means = x[:-tail_length].reshape((-1, subset_size)).mean(axis=1)
        tail_mean = x[-tail_length:].mean()

        res = list()
        for i, mean in enumerate(itertools.chain(means, [tail_mean])):
            start = i * subset_size
            end = min(start + subset_size, len(x))
            res.append({'start': start, 'end': end, 'val': mean})
        return res
    return sub_means
