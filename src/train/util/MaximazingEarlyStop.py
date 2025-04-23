from train.util.EarlyStop import EarlyStop, StepState

class MaximizingEarlyStop(EarlyStop):
    def __init__(self, patience):
        super().__init__(patience)
        self.last_val = -1

    def verify(self, value)-> StepState:
        if self.last_val >= value and abs(self.last_val - value) >= self.significant_diff:
            self.counter += 1
            return StepState.WORSE if self.counter != self.patience else StepState.STOP
        self.last_val = value
        self.counter = 0
        return StepState.BETTER