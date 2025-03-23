import asyncio
from typing import Callable, Coroutine

from async_io import filesystem
from data.text_data_processor.json_data_processor import TwitterJsonDataProcessor
from data.visual_data_processor.resizing_data_processor import ResizingDataProcessor
from data.abstract_dataset_preprocessor import AbstractDatasetPreprocessor
from data.util.file_format_parser import FileFormatParser

class Twitter2015Preprocessor(AbstractDatasetPreprocessor):
    def __init__(self, input_path="../dataset/twitter_2015", output_path="../dataset/preprocessed/twitter_2015"):
        self.input_path = input_path
        self.output_path = output_path
        self.text_processor = TwitterJsonDataProcessor()
        self.image_processor = ResizingDataProcessor()

    async def load_and_transform_dataset(self):
        text_task = asyncio.create_task(self.__load_twitter15_text_dataset(filesystem.save_file))
        #image_task = asyncio.create_task(self.__load_twitter_image_dataset(filesystem.save_file_consumer))
        return await asyncio.gather(text_task)# image_task)

    async def __load_twitter15_text_dataset(self, save_file_consumer: Callable[[str, bytes], Coroutine]):
        text_path = f"{self.input_path}/text"
        queue = asyncio.Queue(maxsize=2 ** 10)
        await asyncio.gather(filesystem.load_directory_contents(text_path, queue),
                             self.__process_twitter15_text_file(queue, save_file_consumer))

    async def __process_twitter15_text_file(self, que: asyncio.Queue, file_writer: Callable[[str, bytes], Coroutine]):
        text_path = f"{self.input_path}/text"
        while True:
            file_wrapper = await que.get()
            if file_wrapper is None:
                break
            file_name = file_wrapper[0]
            file_content = file_wrapper[1]
            parsed_conll = FileFormatParser.parse_t15conll_file(file_content.decode("utf-8"))
            print(parsed_conll)