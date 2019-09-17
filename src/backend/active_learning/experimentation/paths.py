import os
from os import makedirs
from os.path import join

from backend.active_learning.experimentation.configuration.configuration import Configuration


class DirAlreadyExists(RuntimeError):
    def __init__(self):
        return


def create_dirs_decorator_generator(exists_ok=True):

    def create_dirs_decorator(path_getter):

        def decorated_path_getter(*args, get_only=False, **kwargs):
            path = path_getter(*args, **kwargs)
            if not get_only:
                try:
                    makedirs(path, exist_ok=exists_ok)
                except OSError as e:
                    if exists_ok:
                        raise
                    else:
                        raise DirAlreadyExists()
            return path

        return decorated_path_getter

    return create_dirs_decorator


class Paths:
    __RESULTS_DIR = r'experiments_results'

    @staticmethod
    def list_only_dirs(path):
        return [f for f in os.listdir(path) if os.path.isdir(join(path, f))]

    @staticmethod
    def config_as_dir_name(conf: Configuration):
        return 'sub_exp. '\
            f'seed={conf.seed_size}; '\
            f'test={conf.test_size}; '\
            f'C={conf.target_committee_size}; '\
            f'meta={conf.training_method}; '\
            f'#repeats={conf.nb_repeats};'

    @staticmethod
    @create_dirs_decorator_generator()
    def exp_dir(exp_name):
        return join(Paths.__RESULTS_DIR, exp_name)

    @staticmethod
    @create_dirs_decorator_generator()
    def sub_exp_dir(exp_name, sub_exp):
        return join(Paths.exp_dir(exp_name), sub_exp)

    @staticmethod
    @create_dirs_decorator_generator()
    def sub_exp_dir_by_conf(exp_name, conf: Configuration):
        return join(Paths.exp_dir(exp_name), Paths.config_as_dir_name(conf))

    @staticmethod
    @create_dirs_decorator_generator()
    def sub_exp_dataset_dir(exp_name, conf, dataset):
        return join(Paths.sub_exp_dir_by_conf(exp_name, conf), dataset)

    REP_DIR_PREFIX = 'repetition_'
    @staticmethod
    @create_dirs_decorator_generator(exists_ok=False)
    def rep_dir(exp_name, conf, dataset, iter):
        return join(Paths.sub_exp_dataset_dir(exp_name, conf, dataset), f'{Paths.REP_DIR_PREFIX}{iter}')

    AVERAGE_RESULTS_DIR_NAME = 'average_results'
    @staticmethod
    @create_dirs_decorator_generator()
    def avg_res_dir(exp_name, conf, dataset):
        return join(Paths.sub_exp_dataset_dir(exp_name, conf, dataset), Paths.AVERAGE_RESULTS_DIR_NAME)

    CONFIG_FILENAME = 'experiment configuration.txt'
    @staticmethod
    def sub_exp_conf_path(exp_name, conf):
        return join(Paths.sub_exp_dir_by_conf(exp_name, conf), Paths.CONFIG_FILENAME)

    MERGED_RESULTS_DIR_NAME = '_merged_results'
    @staticmethod
    @create_dirs_decorator_generator()
    def merged_results_dir(exp_name, conf):
        return join(Paths.sub_exp_dir_by_conf(exp_name, conf), Paths.MERGED_RESULTS_DIR_NAME)

    @staticmethod
    @create_dirs_decorator_generator()
    def dataset_merged_results_dir(exp_name, conf, dataset):
        return join(Paths.merged_results_dir(exp_name, conf), dataset)

    @staticmethod
    def get_exp_configs(exp_name):
        exp_dir = Paths.exp_dir(exp_name, get_only=True)

        return [
            Configuration.from_json_path(join(exp_dir, d, Paths.CONFIG_FILENAME))
            for d in Paths.list_only_dirs(exp_dir)
        ]

    PIVOT_FILENAME = 'pivot.csv'
    @staticmethod
    def get_pivot_file_path(exp_name):
        return join(Paths.exp_dir(exp_name, get_only=True), Paths.PIVOT_FILENAME)

    @staticmethod
    def rep_idx_from_dir_name(rep_dir):
        return int(rep_dir[len(Paths.REP_DIR_PREFIX):])
