import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import List, Iterable, Union


class ConfigurationParsingError(RuntimeError):
    def __init__(self, *args):
        super().__init__(*args)
        return


@dataclass(repr=False)
class Configuration:
    test_size: int
    seed_size: int
    target_committee_size: int
    training_method: str
    query_measure_points: Iterable
    datasets: List[str]
    nb_repeats: int
    base_estimator: Union[object, Iterable]
    seed_stratified: bool = False

    def __repr__(self):
        line_length = 45
        title = 'Configuration'
        dd = '::'
        d = '.' * ((line_length - len(title)) // 2 - len(dd))
        var_fmt = '{:<24}{}'

        rows = [f'{d}{dd}{title}{dd}{d}'] \
               + [var_fmt.format(var, val) for var, val in self.__dict__.items()] \
               + ['.' * line_length]
        return '\n'.join(rows)

    def to_json_text(self):
        type_repr = lambda be: repr(be)[:repr(be).find('(')]

        d = self.__dict__.copy()
        if type(self.base_estimator) is not str:
            if type(self.base_estimator) is not list:
                d['base_estimator'] = type_repr(self.base_estimator)
            else:
                d['base_estimator'] = [type_repr(be) for be in self.base_estimator]

        d['query_measure_points'] = repr(self.query_measure_points)
        return json.dumps(d)

    @staticmethod
    def from_json_path(json_path):
        with open(json_path, 'r') as f:
            return Configuration.from_json(f.read())

    @staticmethod
    def from_json(configuration_text):
        try:
            return Configuration(**json.loads(configuration_text))
        except JSONDecodeError as e:
            raise ConfigurationParsingError('config error: not a valid json ({})'.format(e))
        except:
            raise ConfigurationParsingError('some error occured; perhaps a wrong set of arguments was given')
