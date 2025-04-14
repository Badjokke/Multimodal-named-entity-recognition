import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Union

from data.data_processor import DataProcessor


class FilteringJsonDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.io_pool_exec = ThreadPoolExecutor(max_workers=5)

    def process_data(self, data: Union[list[str], str]) -> Future[list[str]]:
        assert data is not None, "Data cannot be None"
        return self.io_pool_exec.submit(self.__stem_text if type(data) is list else self.__process_word(data), data)

    def __process_word(self, text: str) -> list[str]:
        return [self.filter_word(text.strip())]

    def __stem_text(self, text: list[str]) -> list[str]:
        return list(map(lambda x: self.filter_word(x.strip()), text))

    def filter_word(self, word: str):
        if len(word) == 0 or self.__is_email_predicate(word) or self.__is_url_predicate(
                word) or self.__is_number_predicate(word) or self.__is_nonword_character_predicate(word):
            return None
        return word

    @staticmethod
    def __is_email_predicate(text: str):
        return re.match("\\w+@\\w+\\.\\w+", text)

    @staticmethod
    def __is_url_predicate(text: str):
        return re.match("https?://\\S+", text)

    @staticmethod
    def __is_number_predicate(text: str):
        return re.match("^\\d+$", text)

    @staticmethod
    def __is_nonword_character_predicate(text: str):
        return re.match("^\\W$", text)
