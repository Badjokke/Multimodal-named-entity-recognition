from abc import ABC, abstractmethod

class EarlyStop(ABC):
    def __init__(self, patience):
        self.last_val = None
        self.patience = patience
        self.counter = 0

    @abstractmethod
    def verify(self, value) -> bool:
        raise NotImplementedError("Cannot invoke abstract method directly")

    def stop(self) -> bool:
        if self.counter == self.patience:
            return True
        return False
