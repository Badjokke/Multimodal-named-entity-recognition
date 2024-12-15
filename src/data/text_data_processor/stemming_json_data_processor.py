from concurrent.futures import ThreadPoolExecutor, Future
import json
from data.data_processor import DataProcessor
from nltk.stem import PorterStemmer

class TwitterStemmingJsonDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.io_pool_exec = ThreadPoolExecutor(max_workers=5)
        self.stemmer = PorterStemmer()

    def process_data(self, data: str) -> Future[tuple[str, str, str]]:
        assert data is not None, "Data cannot be None"
        return self.io_pool_exec.submit(self.__process_json_text, data)

    def __process_json_text(self, text: str) -> tuple[str, str, str]:
        json_value = json.loads(text)
        text = json_value.get('text')
        labels = json_value.get('label')
        related_images = json_value.get('images')
        return json.dumps(self.__stem_text((text[1:-1]).split(","))), json.dumps(related_images), json.dumps(labels)


    def __stem_text(self, text: list[str]) -> str:
        stemmed_text = []
        for i in range(len(text)):
            stemmed_text.append(self.stemmer.stem(text[i]))
        return ",".join(stemmed_text)