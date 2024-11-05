import asyncio
import torch
from src.data.data_processors  import image_to_tensor, parse_twitter_text
import src.async_io.filesystem as filesystem
from concurrent.futures import Future
from typing import Callable

from datasets import load_dataset

input_path = "../dataset/preprocessed/twitter_2017"


# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset


async def load_twitter_dataset():
    return await asyncio.gather(_load_twitter_dataset_text(), _load_twitter_dataset_image())


async def _load_twitter_dataset_text():
    text_path = f"{input_path}/text_preprocessed"
    text_que = asyncio.Queue(100)
    return await asyncio.gather(filesystem.load_directory_contents(text_path, text_que),_load_twitter_text(parse_twitter_text,text_que) )



async def _load_twitter_dataset_image():
    image_path = f"{input_path}/image_preprocessed"
    image_que = asyncio.Queue(100)
    return await asyncio.gather(filesystem.load_directory_contents(image_path, image_que),
                         _transform_images(image_to_tensor, image_que))



async def _transform_images(transform_function: Callable[[bytes], Future[torch.tensor]], queue: asyncio.Queue) -> dict[
    str, torch.tensor]:
    dic = {}
    while True:
        item = await queue.get()
        if item is None:
            break
        result = transform_function(item[1])
        result = result.result(2000)
        dic[item[0]] = result
    return dic


async def _load_twitter_text(text_processor: Callable[[str], Future[tuple[list[str], list[str], list[str]]]],queue: asyncio.Queue) -> list:
    wrapper = []
    while True:
        item = await queue.get()
        if item is None:
            break
        result = text_processor(item[1])
        result = result.result(2000)
        words = result[0]
        images = result[1]
        labels = result[2]
        wrapper.append((words, images, labels))
    return wrapper