import asyncio
from concurrent.futures import Future
from typing import Callable, Union

import torch

from async_io import filesystem
from data.data_processor import DataProcessor
from data.data_processors import image_to_tensor, parse_twitter_text

input_path = "../dataset/preprocessed/twitter_2017"


async def load_twitter_dataset(text_processors: list[DataProcessor] = None,
                               image_processor: list[DataProcessor] = None) -> tuple:
    text_set, image_set = await asyncio.gather(_load_twitter_dataset_text(), _load_twitter_dataset_image())
    return _prepare_twitter_dataset_for_training(text_set, image_set, text_processors, image_processor)


async def dataset_text_only() -> tuple:
    text_set = await asyncio.gather(_load_twitter_dataset_text())
    return _prepare_twitter_dataset_for_training_text(text_set[0])


async def _load_twitter_dataset_text():
    text_path = f"{input_path}/text_preprocessed"
    text_que = asyncio.Queue(2 ** 10)
    result = await asyncio.gather(filesystem.load_directory_contents(text_path, text_que),
                                  _load_twitter_text(parse_twitter_text, text_que))
    return result[1]


async def _load_twitter_dataset_image():
    image_path = f"{input_path}/image_preprocessed"
    image_que = asyncio.Queue(2 ** 10)
    result = await asyncio.gather(filesystem.load_directory_contents(image_path, image_que),
                                  _transform_images(image_to_tensor, image_que))
    return result[1]


async def _transform_images(transform_function: Callable[[bytes], Future[torch.tensor]], queue: asyncio.Queue) -> dict[
    str, torch.tensor]:
    dic = {}
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        result = transform_function(item[1])
        result = result.result(2000)
        dic[item[0]] = result
        queue.task_done()
    return dic


async def _load_twitter_text(text_processor: Callable[[bytes], Future[list[dict[str, str]]]],
                             queue: asyncio.Queue) -> dict:
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
    return wrapper


def _prepare_twitter_dataset_for_training_text(text_set: dict[str, dict[str]]) -> tuple[
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


def _prepare_twitter_dataset_for_training(text_set: dict[str, list[dict[str, list[str]]]],
                                          image_set: dict[str, torch.Tensor],
                                          text_processor=Union[list[DataProcessor], None],
                                          image_processor=Union[list[DataProcessor], None]) -> tuple[
    dict[str, list], dict[str, int], list[int], set[str]]:
    final_dataset = {}
    labels = {}
    class_occurrences = []
    vocabulary = set()
    for key in text_set:
        jsonl_file = text_set[key]
        final_dataset[key] = _process_dataset_part(jsonl_file, labels, text_processor, image_processor, image_set)
        for tpl in final_dataset[key]:
            words, images, lbls = tpl
            for word in words:
                vocabulary.add(word)
            for lbl in lbls:
                class_occurrences.append(lbl)

    return final_dataset, labels, class_occurrences, vocabulary


def _process_dataset_part(jsonl: list[dict[str, list[str]]], labels: dict[str, int],
                          text_data_processor: list[DataProcessor], image_data_processor: list[DataProcessor],
                          image_set: dict[str, torch.Tensor]) -> list[tuple[list[str], torch.Tensor, list[int]]]:
    result = []
    for json in jsonl:
        images = _process_image_refs(image_set, json['image'], image_data_processor)
        text = _apply_data_processor(json['text'], text_data_processor)
        collected_labels = _process_labels(json['label'], labels)
        collected_labels = [label for i, label in enumerate(collected_labels) if text[i] is not None]
        text = list(filter(lambda x: x is not None, text))
        assert len(collected_labels) == len(text), "post filtering of labels and text failed - len diff"
        result.append((text, images, collected_labels))
    return result


def _process_labels(sentence_labels: list[str], labels: dict[str, int]) -> list[int]:
    collected_labels = []
    for label in sentence_labels:
        if label not in labels:
            labels[label] = len(labels)
        collected_labels.append(labels[label])
    return collected_labels


def _process_image_refs(image_set: dict[str, torch.Tensor], image_refs: list[str],
                        image_data_processor: list[DataProcessor], return_first=False) -> torch.Tensor:
    if return_first:
        return image_set[image_refs[0]] if image_data_processor is None else [
            _apply_data_processor(image_set[image_refs[0]], image_data_processor)]
    return torch.stack(list(
        map(lambda ref: image_set[ref], image_refs)) if image_data_processor is None else _map_with_data_processor(
        image_refs, image_data_processor))


def _map_with_data_processor(items: list[Union[str, torch.Tensor]], data_processors: list[DataProcessor]):
    return list(map(lambda item: _apply_data_processor(item, data_processors), items))


def _apply_data_processor(item: Union[list[str], torch.Tensor], data_processors: list[DataProcessor]):
    if data_processors is None or data_processors == []:
        return item
    for processor in data_processors:
        item = processor.process_data(item).result(2000)
    return item
