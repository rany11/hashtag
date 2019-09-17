from src.backend.active_learning.experimentation.experiments.experiment import Experiment
from src.backend.active_learning.experimentation.oracle import Oracle


class MixedVoting(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def simulate_active_learning_scenario(self, configuration: Configuration, strategy: QueryByCommittee, dataset):
        """
        :param configuration:
        :param strategy:
        :param dataset:
        :return:
        """
        oracle = Oracle(dataset.X_unlabeled, dataset.y_oracle)
        committee = Experiment.committee_constructor_dispatch(configuration.training_method)(
            configuration.base_estimator,
            configuration.target_committee_size,
            dataset.X_seed,
            dataset.y_seed,
            dataset.X_all_labels,
            dataset.y_all_labels
        )
        x_axis = configuration.query_measure_points
        for x_current, x_previous in zip(x_axis[1:], x_axis):
            if oracle.is_out_of_data():
                return

            nb_queries = min(x_current - x_previous, oracle.nb_left_samples())
            committee.train()
            committee.acquire_data(
                *oracle.query_update(
                    strategy.query_idx(committee, oracle.X, nb_queries)[:nb_queries]  # add [:nb_queries] just to make sure...
                )
            )
            yield committee
        return
