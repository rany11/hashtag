import os

import pandas as pd
from sklearn.datasets import load_boston, load_breast_cancer, load_diabetes, load_iris, load_linnerud, \
    load_digits, fetch_mldata

from experimentation.dataset.errors import UnknownDatasetError


class Loader:
    __DATASETS_DIR = os.path.join('datasets', 'UCI')

    @staticmethod
    def load_dataset(dataset_name: str):
        def raise_not_found_error():
            raise UnknownDatasetError("dataset \"{}\" not found".format(dataset_name))

        dispatcher = Loader.get_skl_loader_dispatcher()
        dispatcher.update(Loader.get_uci_dataset_dispatcher())

        return dispatcher.get(dataset_name, raise_not_found_error)()

    @staticmethod
    def get_skl_loader_dispatcher():
        skl_dispatcher = {
            'breast-cancer': load_breast_cancer,
            'iris': load_iris,
            'mnist': load_digits,
        }
        skl_dispatcher = {
            name: Loader.skl_loader_unwrap_decorator(skl_dispatcher[name]) for name in skl_dispatcher
        }
        return skl_dispatcher

    @staticmethod
    def skl_loader_unwrap_decorator(dataset_loader):
        def dispatcher():
            dataset = dataset_loader()
            return dataset.data, dataset.target
        return dispatcher

    @staticmethod
    def get_uci_dataset_dispatcher():
        return {
            'avila': Loader.create_uci_loader('Avila', target_column=10, header=None),
            'cars': Loader.create_uci_loader('Car Evaluation', target_column=6, header=None),
            'banknote': Loader.create_uci_loader('banknote authentication', target_column=4, header=None),
            'bank marketing': Loader.create_uci_loader('Bank Marketing', target_column='y', sep=';'),
            'bank marketing full': Loader.create_uci_loader(r'Bank Marketing\full', target_column='y', sep=';'),
            'cnae': Loader.create_uci_loader('CNAE-9', target_column=0, header=None),
            'wilt': Loader.create_uci_loader('wilt', target_column='class'),
            'heart': Loader.create_uci_loader('Heart Disease UCI', target_column='target'),
            'credit': Loader.create_uci_loader('Credit Card Fraud Detection', target_column='Class', drop=["Time"]),
            'yeast': Loader.create_uci_loader('yeast', target_column=9, drop=[0], header=None),
        }

    @staticmethod
    def create_uci_loader(*args, **kwargs):
        return lambda: Loader.load_uci_dataset(*args, **kwargs)

    @staticmethod
    def load_uci_dataset(dataset_name, target_column, drop=None, *args, **kwargs):
        # TODO: add option to load pre-made test data
        dataset_path = os.path.join(Loader.__DATASETS_DIR, dataset_name, 'data')

        if not os.path.exists(dataset_path):
            raise UnknownDatasetError("dataset \"{}\" not found".format(dataset_name))

        df = pd.read_csv(dataset_path, *args, **kwargs)

        if target_column not in df.columns:
            raise RuntimeError('target columns "{}" not in columns {} for dataset "{}"'.format(target_column,
                                                                                               df.columns,
                                                                                               dataset_name))
        data_columns = list(df.columns)
        if drop is not None:
            data_columns = list(set(data_columns).difference(drop))

        data_columns.remove(target_column)
        target = df[target_column].values  # last column in the csv is the target
        data = df[data_columns].values     # other columns are the features
        return data, target
