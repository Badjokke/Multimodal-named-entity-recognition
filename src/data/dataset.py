import asyncio
from concurrent.futures import Future
from typing import Callable, Coroutine

from datasets import load_dataset


input_path = "../../dataset/twitter_2017"


# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset


async def load_twitter_dataset(text_processor: Callable[[str], Future[str]],
                               image_processor: Callable[[bytes, str], Future[bytes]]):
    pass

