from abc import ABC, abstractmethod


class AbstractDatasetPreprocessor(ABC):
    @abstractmethod
    async def load_and_transform_dataset(self):
        raise NotImplementedError("Can not directly invoke load_twitter_dataset")
