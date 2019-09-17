from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

from backend.active_learning.experimentation.configuration.configuration import Configuration


@dataclass
class ConfigFactory(Configuration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def generate(self, group_datasets=False):
        fields = {'seed_size': self.seed_size,
                  'target_committee_size': self.target_committee_size,
                  'training_method': self.training_method}

        changing_parameters = {k: v for k, v in fields.items() if isinstance(v, Iterable)}

        if not group_datasets:
            self.datasets = [[d] for d in self.datasets]
            changing_parameters['datasets'] = self.datasets

        for values in product(*changing_parameters.values()):
            d = self.__dict__.copy()
            d.update({k: v for k, v in zip(changing_parameters.keys(), values)})
            yield Configuration(**d)

    def generate_list(self):
        return [conf for conf in self.generate()]


if __name__ == '__main__':
    for c in ConfigFactory(100, [10, 20, 30], False, [3, 7, 11], ['bagging', 'adaboost'],
                           range(100), ['heart', 'yeast'], 15, 'clf').generate():
        print(c)
