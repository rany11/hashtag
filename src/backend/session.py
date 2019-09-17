from PIL import Image
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

from backend.active_learning.committee_utils.committee import Committee
from backend.active_learning.experimentation.exporters.implementations import Accuracy, accuracy
from src.backend.active_learning.active_learning.vote_entropy import VoteEntropy
from src.backend.active_learning.experimentation.configuration.configuration import Configuration
from src.backend.active_learning.experimentation.dataset.dataset import DataSet
from src.backend.active_learning.experimentation.dataset.loader import Loader
from src.backend.active_learning.experimentation.experiments.experiment import Experiment
from src.backend.active_learning.experimentation.oracle import Oracle
import numpy as np


def show_mnist_image(image):
    Image.fromarray(image * 16).resize((256, 256)).show()
    return


class Session:
    def __init__(self, dataset_name):
        data, target = Loader.load_dataset(dataset_name)
        conf = Configuration(
            test_size=300,
            seed_size=10,
            target_committee_size=3,
            training_method='bagging',
            query_measure_points=[],    # dummy
            datasets=[],                # dummy
            nb_repeats=0,               # dummy
            base_estimator=LogisticRegression(multi_class='auto', solver='liblinear'),
            seed_stratified=True
        )
        self.ds = DataSet(data, target, conf)
        self.oracle = Oracle(self.ds.X_unlabeled, self.ds.y_oracle)
        self.strategy = VoteEntropy()
        self.committee = Experiment.committee_constructor_dispatch(conf.training_method)(
            conf.base_estimator,
            conf.target_committee_size,
            self.ds.X_seed,
            self.ds.y_seed,
            self.ds.X_all_labels,
            self.ds.y_all_labels
        )
        self.committee.train()
        return

    def next_id(self):
        if self.oracle.is_out_of_data():
            raise RuntimeError('out of data')

        ids = self.strategy.query_idx(self.committee, self.oracle.X, nb_queries=1)
        print(f'ids is {ids}')
        return ids[0]

    def take_label(self, sample_id, label):
        label = int(label)
        X, y = self.oracle.query_update(np.array([sample_id]))

        if label != y[0]:
            print(f'user gave wrong label {label} (actual: {y[0]})')
        else:
            print(f'user is correect! ({label})')
        self.committee.acquire_data(X, np.array([label]))
        self.committee.train()
        return

    def predict(self, sample_id):
        votes = self.committee.vote([self.oracle.X[sample_id]])
        elements, counts = np.unique(votes, return_counts=True)
        return elements[np.argmax(counts)]

    def estimate_accuracy(self):
        return accuracy(Committee.majority(self.committee.vote(self.ds.X_test)), self.ds.y_test)
