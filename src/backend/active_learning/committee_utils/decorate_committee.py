from committee_utils.committee import Committee


class CommitteeDecorate(Committee):
    def train(self):
        return self.__decorate()

    def __decorate(self):
        raise NotImplementedError()
