from abc import ABC, abstractmethod
from concurrent.futures.thread import ThreadPoolExecutor

from data.data_processor import DataProcessor


class AbstractDatasetLoader(ABC):
    def __init__(self, input_path: str, text_processors: list[DataProcessor] = None, image_processors: list[DataProcessor] = None):
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.input_path = input_path
        self.text_processors = text_processors
        self.image_processors = image_processors

    @abstractmethod
    async def load_dataset(self) -> tuple:
        raise NotImplementedError("Cannot directly call load_twitter_dataset")
