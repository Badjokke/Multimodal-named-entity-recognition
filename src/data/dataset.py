import asyncio
from concurrent.futures import Future
from typing import Callable

import torch
from data.data_processors import image_to_tensor, parse_twitter_text
from datasets import load_dataset
from async_io import filesystem
input_path = "../dataset/preprocessed/twitter_2017"


# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset


async def load_twitter_dataset() -> tuple:
    text_set, image_set = await asyncio.gather(_load_twitter_dataset_text(), _load_twitter_dataset_image())
    return _prepare_twitter_dataset_for_training(text_set, image_set)


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


async def _load_twitter_text(text_processor: Callable[[str], Future[tuple[list[str], list[str], list[str]]]],
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


# todo threads optimisation
# todo consumes way too much memory
# this just feels wrong, rewrite
def _prepare_twitter_dataset_for_training(text_set: dict[str, dict[str]], image_set: dict[str, torch.Tensor]) -> tuple[
    dict[str, list], dict[str, int]]:
    final_dataset = {}
    labels = {}
    label_id = 0
    for key in text_set:
        jsonl_file = text_set[key]
        final_dataset[key] = []
        for json in jsonl_file:
            image_refs = json['image']
            sentence_labels = json['label']
            image_tensors = []
            for label in sentence_labels:
                if label not in labels:
                    labels[label] = label_id
                    label_id += 1
            for image in image_refs:
                image_tensor = image_set[image]
                if image_tensor is None:
                    print(f"Tensor for image ref: {image} missing!")
                    continue
                image_tensors.append(image_tensor / 255)
            final_dataset[key].append((json['text'], torch.stack(image_tensors, dim=0), sentence_labels))
    return final_dataset, labels
