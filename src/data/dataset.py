import asyncio
import time
from concurrent.futures import Future
from typing import Callable, Coroutine

from datasets import load_dataset

from src.async_io import filesystem
from src.data.data_processors import process_twitter2017_image, process_twitter2017_text

path = "../../dataset/twitter_2017"


# Load the databricks dataset from Hugging Face
def load_hf_dataset():
    dataset = load_dataset("conll2003", trust_remote_code=True)
    return dataset


async def load_twitter_dataset(text_processor: Callable[[str], Future[str]],
                               image_processor: Callable[[bytes, str], Future[bytes]]):
    text_task = asyncio.create_task(__load_twitter_text_dataset(text_processor, filesystem.save_file))
    await text_task
    # image_task = asyncio.create_task(__load_twitter_image_dataset(image_processor, filesystem.save_file))
    # return await asyncio.gather(text_task, image_task)


async def __load_twitter_image_dataset(image_processor: Callable[[bytes, str], Future[bytes]],
                                       file_writer: Callable[[str, bytes], Coroutine]):
    in_path = f"{path}/images"
    unfinished_queue = asyncio.Queue(maxsize=500)

    await asyncio.gather(filesystem.load_directory_contents(in_path, unfinished_queue),
                         __process_image(image_processor, unfinished_queue, file_writer))


async def __process_image(image_processor: Callable[[bytes, str], Future[bytes]],
                          queue: asyncio.Queue[tuple[str, bytes]],
                          file_writer: Callable[[str, bytes], Coroutine]):
    out_path = f"{path}/image_preprocessed"
    while True:
        item = await queue.get()
        if item is None:
            break
        bin_data = item[1]
        file_name = item[0]
        future_result = image_processor(bin_data, file_name.split('.')[-1])
        result = future_result.result(2000)
        await file_writer(f"{out_path}/{file_name}", result)
        # todo
        # future_result.add_done_callback(lambda future: file_writer(f"{out_path}/{file_name}",future.result()))


async def __load_twitter_text_dataset(text_processor: Callable[[str], Future[str]],
                                      file_writer: Callable[[str, bytes], Coroutine]):
    in_path = f"{path}/text"
    unfinished_queue = asyncio.Queue(maxsize=500)
    await asyncio.gather(filesystem.load_directory_contents_generator(in_path, unfinished_queue),
                         __process_text(text_processor, unfinished_queue, file_writer))


async def __process_text(text_processor: Callable[[str], Future[str]], queue: asyncio.Queue[tuple[str, str]],
                         file_writer: Callable[[str, bytes], Coroutine]):
    out_path = f"{path}/text_processed"
    # TODO file appender
    #
    file_buffer = ""
    while True:
        item = await queue.get()
        if item is None:
            break
        bin_data = item[1]
        status = item[0]
        if bin_data == "EOF":
            file_name = status
            await file_writer(f"{out_path}/{file_name}", str.encode(file_buffer, encoding="utf-8"))
            file_buffer = ""
            continue
        future_result = text_processor(bin_data)
        result = future_result.result(2000)
        file_buffer += result


if __name__ == "__main__":
    print("Loading dataset")
    start = time.time()
    asyncio.run(load_twitter_dataset(process_twitter2017_text, process_twitter2017_image), debug=True)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")
