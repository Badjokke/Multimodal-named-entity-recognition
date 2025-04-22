from abc import ABC, abstractmethod
from enum import Enum
class StepState(Enum):
    BETTER = 0,
    WORSE = 1,
    STOP = 2
class EarlyStop(ABC):
    def __init__(self, patience):
        self.last_val = None
        self.patience = patience
        self.counter = 0
        self.significant_diff = 2

    @abstractmethod
    def verify(self, value) -> StepState:
        raise NotImplementedError("Cannot invoke abstract method directly")
