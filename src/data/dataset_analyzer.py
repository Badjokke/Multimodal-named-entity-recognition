from typing import Union


class DatasetAnalyzer:
    def __init__(self, dataset: dict[str, list], class_occurrences: list[int], labels: dict[str, int], vocabulary: dict[str, int]):
        assert "train" in dataset and "test" in dataset and "val" in dataset, "dataset must have keys: train test and val"
        assert len(dataset["train"][0]) == 3 == len(dataset["test"][0]) == len(dataset["val"][0]), "each subset must have text, image, label structure"
        self.dataset = dataset
        self.class_occurrences = class_occurrences
        self.labels = labels
        self.vocabulary = vocabulary

    def get_dataset_label_count(self) -> dict:
        return self.__get_label_count_from_occurrences(self.labels, self.class_occurrences)

    def get_train_subset_stats(self)-> dict:
        return self.__stats(self.dataset['train'],{value: key for key, value in self.labels.items()})

    def get_validation_subset_stats(self)-> dict:
        return self.__stats(self.dataset['val'],{value: key for key, value in self.labels.items()})

    def get_test_subset_stats(self)-> dict:
        return self.__stats(self.dataset['test'],{value: key for key, value in self.labels.items()})

    def get_label_names(self):
        return self.labels.keys()

    def get_unique_token_count(self) -> int:
        return len(self.vocabulary)

    def get_train_val_test_sentence_count(self) -> tuple[int, int, int]:
        return len(self.dataset["train"]), len(self.dataset["val"]), len(self.dataset["test"])

    @staticmethod
    def __get_label_count_from_occurrences(labels:dict[str, int], occurrences: list[int]) -> dict[str,int]:
        label_id_to_str = {value: key for key, value in labels.items()}
        histogram = {key: 0 for key in labels.keys()}
        for label_id in occurrences:
            histogram[label_id_to_str[label_id]] += 1
        return histogram

    @staticmethod
    def __stats(data:list[tuple[list[str], list[int], list[int]]], label_id_to_label: dict[int, str]) -> dict[str, dict|int]:
        label_histogram = {key: 0 for key in label_id_to_label.values()}
        token_count = 0
        unique_tokens = set()
        image_count = 0
        sample_size = len(data)
        for i in range(sample_size):
            sample = data[i]
            words = sample[0]
            image_count += len(sample[1])
            labels = sample[2]
            token_count += len(words)
            for word in words:
                unique_tokens.add(word)
            for label in labels:
                label_histogram[label_id_to_label[label]] += 1
        return {"hist":label_histogram, "token_count":token_count, "unique_tokens":len(unique_tokens), "sentence_count":sample_size,"image_count":image_count, "labels_total":sum(label_histogram.values())}