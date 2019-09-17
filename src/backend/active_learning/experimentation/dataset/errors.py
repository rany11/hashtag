class UnknownDatasetError(RuntimeError):
    def __init__(self, msg="dataset not found"):
        """
        :param msg: error message
        """
        self.msg = msg
        return

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.msg
