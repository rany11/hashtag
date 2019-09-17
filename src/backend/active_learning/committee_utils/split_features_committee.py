from committee_utils.committee import Committee


class CommitteeSplitFeatures(Committee):
    def train(self):
        return self.__split_features()

    def __split_features(self):
        raise NotImplementedError()
