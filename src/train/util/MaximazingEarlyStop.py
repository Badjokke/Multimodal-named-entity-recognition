from train.util.EarlyStop import EarlyStop


class MaximizingEarlyStop(EarlyStop):
    def __init__(self, patience):
        super().__init__(patience)

    def verify(self, value)->bool:
        if self.last_val > value:
            self.counter += 1
        return self.stop()