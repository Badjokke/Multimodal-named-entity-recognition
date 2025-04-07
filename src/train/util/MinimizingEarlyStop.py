from train.util.EarlyStop import EarlyStop


class MinimizingEarlyStop(EarlyStop):
    def __init__(self):
        super().__init__()

    def validate(self, value)->bool:
        if self.last_val < value:
            self.counter += 1
        return self.stop()
