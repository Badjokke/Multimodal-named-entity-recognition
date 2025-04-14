from nltk.stem import PorterStemmer

from data.text_data_processor.FilteringJsonDataProcessor import FilteringJsonDataProcessor


class StemmingJsonDataProcessor(FilteringJsonDataProcessor):
    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()

    def filter_word(self, word: str):
        filtered_word = super().filter_word(word)
        return self.stemmer.stem(filtered_word) if filtered_word else None
