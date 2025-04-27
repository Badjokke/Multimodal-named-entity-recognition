import asyncio
from concurrent.futures import Future
from typing import Callable, Union

import torch
from async_io import filesystem
from data.abstract_dataset_loader import AbstractDatasetLoader
from data.data_processor import DataProcessor
from data.data_processors import image_to_tensor, parse_twitter_text
from sklearn.model_selection import train_test_split


class JsonlDatasetLoader(AbstractDatasetLoader):
    def __init__(self, input_path: str = "../dataset/preprocessed/twitter_2017",
                 lightweight=False, include_parent_dir=False, custom_split=False,
                 text_processors: list[DataProcessor] = None, image_processors: list[DataProcessor] = None):
        """
        class is responsible for loading twitter 2017 in it's raw form - jsonl files and jpges
        and process it to
        """
        super().__init__(input_path, text_processors, image_processors)
        self.lightweight = lightweight
        self.include_parent_dir = include_parent_dir
        self.custom_split = custom_split

    async def load_dataset(self) -> tuple:
        text_set, image_set = await asyncio.gather(self.__load_twitter_dataset_text(),
                                                   self.__load_twitter_dataset_image())
        return self.__prepare_twitter_dataset_for_training(text_set, image_set)

    async def dataset_text_only(self) -> tuple:
        text_set = await asyncio.gather(self.__load_twitter_dataset_text())
        return self.__prepare_twitter_dataset_for_training_text(text_set[0])

    async def __load_twitter_dataset_text(self):
        text_path = f"{self.input_path}/text_preprocessed"
        text_que = asyncio.Queue(2 ** 8)
        result = await asyncio.gather(filesystem.load_directory_contents(text_path, text_que),
                                      self.__load_twitter_text(parse_twitter_text, text_que,
                                                               train_test_split=self.custom_split))
        return result[1]

    async def __load_twitter_dataset_image(self):
        image_path = f"{self.input_path}/image_preprocessed"
        image_que = asyncio.Queue(2 ** 8)
        result = await asyncio.gather(
            filesystem.load_directory_contents(image_path, image_que, include_parent_dir=self.include_parent_dir),
            self.__transform_images(image_to_tensor, image_que, self.lightweight))
        return result[1]

    @staticmethod
    async def __transform_images(transform_function: Callable[[bytes], Future[torch.tensor]], queue: asyncio.Queue,
                                 lightweight=False) -> \
            dict[
                str, torch.tensor]:
        dic = {}
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            if lightweight:
                dic[item[0]] = 1
            else:
                result = transform_function(item[1])
                result = result.result(2000)
                dic[item[0]] = result
            queue.task_done()
        return dic

    @staticmethod
    async def __load_twitter_text(text_processor: Callable[[bytes], Future[list[dict[str, str]]]],
                                  queue: asyncio.Queue, train_test_split=False) -> dict:
        wrapper = {}
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            result = text_processor(item[1])
            result = result.result(2000)
            queue.task_done()
            wrapper[item[0].split("_")[-1].split(".")[0]] = result
        return wrapper if not train_test_split else JsonlDatasetLoader.custom_train_test_val_split(wrapper['dataset'])

    @staticmethod
    def custom_train_test_val_split(dataset):
        train, test = train_test_split(dataset, test_size=0.2, random_state=142, shuffle=True)
        train, val = train_test_split(dataset, test_size=0.1, random_state=142, shuffle=True)
        return {"train": train, "val": val, "test": test}

    @staticmethod
    def __prepare_twitter_dataset_for_training_text(text_set: dict[str, dict[str]]) -> tuple[
        dict[str, list], dict[str, int]]:
        final_dataset = {}
        labels = {}
        label_id = 0
        for key in text_set:
            jsonl_file = text_set[key]
            final_dataset[key] = []
            for json in jsonl_file:
                sentence_labels = json['label']
                sentence_labels_id = []
                for label in sentence_labels:
                    if label not in labels:
                        labels[label] = label_id
                        label_id += 1
                    sentence_labels_id.append(labels[label])
                final_dataset[key].append((json['text'], torch.tensor(sentence_labels_id)))
        return final_dataset, labels

    def __prepare_twitter_dataset_for_training(self, text_set: dict[str, list[dict[str, list[str]]]],
                                               image_set: dict[str, torch.Tensor]) -> tuple[
        dict[str, list], dict[str, int], list[int], dict[str, str]]:
        final_dataset = {}
        labels = {}
        class_occurrences = []
        vocabulary = {}
        word_id = 0
        for key in text_set:
            jsonl_file = text_set[key]
            final_dataset[key] = self.__process_dataset_part(jsonl_file, labels, image_set)
            for tpl in final_dataset[key]:
                words, images, lbls = tpl
                for word in words:
                    if word not in vocabulary:
                        vocabulary[word] = word_id
                        word_id += 1
                for lbl in lbls:
                    class_occurrences.append(lbl)

        return final_dataset, labels, class_occurrences, vocabulary

    def __process_dataset_part(self, jsonl: list[dict[str, list[str]]], labels: dict[str, int],
                               image_set: dict[str, torch.Tensor]) -> list[tuple[list[str], torch.Tensor, list[int]]]:
        result = []
        for json in jsonl:
            image_refs = json['image'] if "image" in json else json["images"]
            images = self.__process_image_refs(image_set, image_refs)
            # image is missing from dataset, skip
            if images is None:
                continue
            text = self.__apply_data_processor(json['text'], self.text_processors)
            collected_labels = self.__process_labels(json['label'], labels)
            collected_labels = [label for i, label in enumerate(collected_labels) if text[i] is not None]
            text = list(filter(lambda x: x is not None, text))
            # text is corrupt or filtered out if data_processors are not none; skip
            if len(text) == 0:
                continue
            assert len(collected_labels) == len(text), "post filtering of labels and text failed - len diff"
            result.append((text, images, collected_labels))
        return result

    @staticmethod
    def __process_labels(sentence_labels: list[str], labels: dict[str, int]) -> list[int]:
        collected_labels = []
        for label in sentence_labels:
            if label not in labels:
                labels[label] = len(labels)
            collected_labels.append(labels[label])
        return collected_labels

    def __process_image_refs(self, image_set: dict[str, torch.Tensor], image_refs: list[str]) -> torch.Tensor:
        # just return 1 as placeholder for image since we are not loading them into memory
        if self.lightweight:
            return [1] * len(image_refs)
        image_refs = list(filter(lambda ref: ref in image_set, image_refs))
        if len(image_refs) == 0:
            return None
        return torch.stack(list(
            map(lambda ref: image_set[ref],
                image_refs)) if self.image_processors is None else self.__map_with_data_processor(
            image_refs, self.image_processors))

    @staticmethod
    def __map_with_data_processor(items: list[Union[str, torch.Tensor]], data_processors: list[DataProcessor]):
        return list(map(lambda item: JsonlDatasetLoader.__apply_data_processor(item, data_processors), items))

    @staticmethod
    def __apply_data_processor(item: Union[list[str], torch.Tensor], data_processors: list[DataProcessor]):
        if data_processors is None or data_processors == []:
            return item
        for processor in data_processors:
            item = processor.process_data(item).result(2000)
        return item
