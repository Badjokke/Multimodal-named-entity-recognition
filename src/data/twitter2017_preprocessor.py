import asyncio
from typing import Callable, Coroutine

from async_io import filesystem
from data.text_data_processor.json_data_processor import TwitterJsonDataProcessor
from data.visual_data_processor.resizing_data_processor import ResizingDataProcessor
from data.abstract_dataset_preprocessor import AbstractDatasetPreprocessor


class Twitter2017Preprocessor(AbstractDatasetPreprocessor):
    def __init__(self, input_path="../dataset/twitter_2017", output_path="../dataset/preprocessed/twitter_2017"):
        self.input_path = input_path
        self.output_path = output_path
        self.text_processor = TwitterJsonDataProcessor()
        self.image_processor = ResizingDataProcessor()

    async def load_and_transform_dataset(self):
        text_task = asyncio.create_task(self.__load_twitter_text_dataset(filesystem.save_file))
        image_task = asyncio.create_task(self.__load_twitter_image_dataset(filesystem.save_file_consumer))
        return await asyncio.gather(text_task, image_task)

    async def __load_twitter_image_dataset(self, file_writer: Callable[[asyncio.Queue[tuple[str, bytes]]], Coroutine]):
        in_path = f"{self.input_path}/images"
        unfinished_queue = asyncio.Queue(maxsize=500)

        await asyncio.gather(filesystem.load_directory_contents(in_path, unfinished_queue),
                             self.__process_image(unfinished_queue, file_writer))

    async def __process_image(self, queue: asyncio.Queue[tuple[str, bytes]],
                              file_writer: Callable[[asyncio.Queue[tuple[str, bytes]]], Coroutine]):
        out_path = f"{self.output_path}/image_preprocessed"
        write_queue = asyncio.Queue(maxsize=2 ** 12)
        asyncio.create_task(file_writer(write_queue))
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                await write_queue.put(None)
                break
            bin_data = item[1]
            file_name = item[0]
            future_result = self.image_processor.process_data(bin_data).result(2000)
            queue.task_done()
            await write_queue.put((f"{out_path}/{file_name}", future_result))

        await write_queue.join()

    async def __load_twitter_text_dataset(self, file_writer: Callable[[str, bytes], Coroutine]):
        in_path = f"{self.input_path}/text"
        unfinished_queue = asyncio.Queue(maxsize=500)
        await asyncio.gather(filesystem.load_directory_contents_generator(in_path, unfinished_queue),
                             self.__process_text(unfinished_queue, file_writer))

    async def __process_text(self,
                             queue: asyncio.Queue[tuple[str, str]],
                             file_writer: Callable[[str, bytes], Coroutine]):
        out_path = f"{self.output_path}/text_preprocessed"
        string_buffer = []
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            bin_data = item[1]
            status = item[0]
            if bin_data == "EOF":
                file_name = status
                await file_writer(f"{out_path}/{file_name}", str.encode("".join(string_buffer), encoding="utf-8"))
                string_buffer = []
                continue
            self.text_processor.process_data(bin_data).add_done_callback(
                lambda future: (
                    string_buffer.append(
                        f"{{\"text\":{future.result()[0]},\"image\":{future.result()[1]},\"label\":{future.result()[2]}}}\n"),
                    queue.task_done()
                )
            )
