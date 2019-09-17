import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from backend.active_learning.experimentation.configuration.configuration import Configuration
from backend.active_learning.experimentation.dataset.utils import shuffle_dataset, empty_array_from_template


class DataSet:
    def __init__(self, X, y, config: Configuration):
        # for labels with continuous values,
        # but they cause other problems later on
        # y = preprocessing.LabelEncoder().fit_transform(y)

        seed_size, test_size = config.seed_size, config.test_size
        X, y = shuffle_dataset(X, y)

        self.all_labels = np.unique(y)
        self.nb_labels = len(self.all_labels)

        if config.seed_stratified:
            self.X_seed, self.y_seed, X, y = DataSet.split_part_same_distrib(X, y, seed_size)
            if self.X_seed.shape[0] > seed_size:
                raise RuntimeError("seed set larger than seed_size; please choose a larger seed size")

            self.X_all_labels, self.y_all_labels, \
            self.X_seed, self.y_seed = DataSet.split_all_labels(self.X_seed, self.y_seed)
        else:
            self.X_all_labels, self.y_all_labels, X, y = DataSet.split_all_labels(X, y)
            print(seed_size)
            print(self.nb_labels)
            seed_size -= self.nb_labels  # X_all_labels is included in the seed set
            print(seed_size)
            if seed_size < 0:
                raise RuntimeError("seed_size smaller than the number of labels; please choose a larger seed size")
            elif seed_size == 0:
                self.X_seed, self.y_seed = empty_array_from_template(X), empty_array_from_template(y)
            else:
                self.X_seed, self.y_seed, X, y = DataSet.split_part(X, y, seed_size)

        self.X_test, self.y_test, X, y = DataSet.split_part_same_distrib(X, y, test_size)
        self.X_unlabeled, self.y_oracle = X, y

        self.data_summary = DataSet.summarize_data(self.all_labels,
                                                   self.y_all_labels, self.y_seed, self.y_oracle, self.y_test,
                                                   self.X_all_labels, self.X_seed, self.X_unlabeled, self.X_test)

        self.test_label_idx = []
        self.test_by_label = []
        self.testset_class_weights = dict()
        for label in self.all_labels:
            idx = (self.y_test == label)
            self.test_label_idx.append(idx)
            self.test_by_label.append(self.X_test[idx])
            self.testset_class_weights[label] = idx.astype(np.int).sum() / self.y_test.shape[0]
        return

    @staticmethod
    def split_all_labels(X, y):
        all_labels = np.unique(y)
        idx = np.ndarray((all_labels.shape[0],), dtype=np.int)
        for i, label in enumerate(np.unique(y)):
            idx[i] = np.where(y == label)[0][0]
        rest_idx = np.setdiff1d(np.arange(X.shape[0], dtype=np.int), idx)

        X_all_labels, y_all_labels = X[idx], y[idx]
        X_rest, y_rest = X[rest_idx], y[rest_idx]

        return X_all_labels, y_all_labels, X_rest, y_rest

    @staticmethod
    def split_part(X, y, part, *args, **kwargs):
        X_rest, X_part, \
        y_rest, y_part = train_test_split(X, y, test_size=part, *args, **kwargs)
        return X_part, y_part, X_rest, y_rest

    @staticmethod
    def split_part_same_distrib(X, y, part, *args, **kwargs):
        all_labels, counts = np.unique(y, return_counts=True)  # classes histogram

        # number of samples to take from each class
        take_counts = (part * counts / counts.sum()).astype(int)
        take_counts[take_counts == 0] = 1

        # number of samples we miss to reach the requested number of samples ('part')
        leftover = part - take_counts.sum()

        # for the main loop
        def split_part_and_append(X1, y1, X2, y2, X_original, y_original, part):
            X_new1, y_new1, X_new2, y_new2 = DataSet.split_part(X_original, y_original, part)
            return [np.append(old, new, axis=0) for old, new in zip([X1, y1, X2, y2],
                                                                    [X_new1, y_new1, X_new2, y_new2])]

        # init arrays
        X_part, y_part = empty_array_from_template(X), empty_array_from_template(y)
        X_rest, y_rest = empty_array_from_template(X), empty_array_from_template(y)

        # main loop: draw the specified number of samples for each class
        for l, count in zip(all_labels, take_counts):
            idx = (y == l)
            X_part, y_part, X_rest, y_rest = split_part_and_append(X_part, y_part,
                                                                   X_rest, y_rest,
                                                                   X[idx], y[idx],
                                                                   count)
        # shuffle the rest of the data...
        X_rest, y_rest = shuffle_dataset(X_rest, y_rest)

        # in case that the requested size is not fulfilled because [part]*[actual # of samples] is rounded down too much
        # we draw some random samples from the rest of the data
        if leftover != 0:
            X_part_left, y_part_left, X_rest, y_rest = DataSet.split_part(X_rest, y_rest, leftover)
            X_part, y_part = shuffle_dataset(np.append(X_part, X_part_left, axis=0),
                                             np.append(y_part, y_part_left, axis=0))
        return X_part, y_part, X_rest, y_rest

    @staticmethod
    def summarize_data(all_labels,
                       y_all_labels, y_seed, y_oracle, y_test,
                       X_all_labels, X_seed, X_unlabeled, X_test):
        fmt_string = """{}
        seed:           {}
        active pool:    {}
        test:           {}    
        X_all_labels:   {}
        """
        summary = list()
        summary.append(fmt_string.format('Subsets Total Size (and shape):',
                                         X_seed.shape,
                                         X_unlabeled.shape,
                                         X_test.shape,
                                         X_all_labels.shape))

        for l in all_labels:
            summary.append(
                fmt_string.format(f'Num of samples of label {l}:', *[(y == l).astype(np.int).sum()
                                                                     for y in [y_seed, y_oracle, y_test, y_all_labels]])
            )

        summary = '\n'.join(summary)
        return summary


if __name__ == '__main__':
    conf = Configuration(
        test_size=6,
        seed_size=3,
        target_committee_size=3,
        training_method='bagging',
        query_measure_points=[],  # dummy
        datasets=[],  # dummy
        nb_repeats=0,  # dummy
        base_estimator=LogisticRegression(multi_class='auto', solver='liblinear'),
        seed_stratified=True
    )
    y = np.concatenate([np.arange(3) for _ in range(5)])
    x = np.ndarray((3*5, 2))
    x[:, 0] = y
    x[:, 1] = np.arange(15)

    print(x)
    print(y)
    ds = DataSet(x, y, conf)
    print(ds.X_unlabeled)
    print(ds.y_oracle)
    print('-' * 10)
    print(ds.X_all_labels)
    print(ds.y_all_labels)
    print('-' * 10)
    print(ds.X_test)
    print(ds.y_test)
    print('-' * 10)
    print(ds.X_seed)
    print(ds.y_seed)
