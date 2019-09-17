from sklearn.ensemble import VotingClassifier

from src.backend.active_learning.committee_utils.standard_ensemble_methods import StandardEnsembleCommittee


class CommitteeDiffClfs(StandardEnsembleCommittee):
    def construct_skl_trainer(self):
        return VotingClassifier([(f'clf{i}', clf) for i, clf in enumerate(self.base_estimator)], voting='hard')
