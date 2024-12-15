from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor


class DataProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def process_data(self, data: Union[bytes, str, Tensor]):
        raise NotImplementedError('Method process_data must be implemented')