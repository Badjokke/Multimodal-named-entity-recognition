import json
from concurrent.futures import ThreadPoolExecutor, Future

from data.data_processor import DataProcessor


class TwitterJsonDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.io_pool_exec = ThreadPoolExecutor(max_workers=5)

    def process_data(self, data: str) -> Future[tuple[str, str, str]]:
        assert data is not None, "Data cannot be None"
        return self.io_pool_exec.submit(self.__process_json_text, data)

    @staticmethod
    def __process_json_text(text: str) -> tuple[str, str, str]:
        json_value = json.loads(text)
        text = json_value.get('text')
        labels = json_value.get('label')
        related_images = json_value.get('images')
        return json.dumps(text), json.dumps(related_images), json.dumps(labels)
