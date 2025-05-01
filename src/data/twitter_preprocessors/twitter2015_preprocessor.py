import asyncio
import json
from functools import reduce
from typing import Callable, Coroutine

from async_io import filesystem
from data.abstract_dataset_preprocessor import AbstractDatasetPreprocessor
from data.label_data_processor.label_mapping_processor import LabelMappingProcessor
from data.util.file_format_parser import FileFormatParser
from data.visual_data_processor.resizing_data_processor import ResizingDataProcessor
from util.directories_util import DirectoryUtil


class Twitter2015Preprocessor(AbstractDatasetPreprocessor):
    def __init__(self, input_path="../dataset/twitter_2015", output_path="../dataset/preprocessed/twitter_2015"):
        self.input_path = input_path
        self.output_path = output_path
        self.image_processor = ResizingDataProcessor()
        #maps to t17 classes
        self.label_processor = LabelMappingProcessor({"I-OTHER": "I-MIS", "B-OTHER": "B-MIS"})
        DirectoryUtil.create_preprocessed_dataset_directories(self.output_path)

    async def load_and_transform_dataset(self):
        text_task = asyncio.create_task(self.__load_twitter15_text_dataset(filesystem.save_file))
        image_task = asyncio.create_task(self.__load_twitter_image_dataset(filesystem.save_file_consumer))
        return await asyncio.gather(text_task, image_task)  # image_task)

    async def __load_twitter15_text_dataset(self, save_file_consumer: Callable[[str, bytes], Coroutine]):
        text_path = f"{self.input_path}/text"
        queue = asyncio.Queue(maxsize=2 ** 10)
        await asyncio.gather(filesystem.load_directory_contents(text_path, queue),
                             self.__process_twitter15_text_file(queue, save_file_consumer))

    async def __process_twitter15_text_file(self, que: asyncio.Queue, file_writer: Callable[[str, bytes], Coroutine]):
        while True:
            file_wrapper = await que.get()
            if file_wrapper is None:
                break
            file_name = file_wrapper[0].split(".")[0]
            file_content = file_wrapper[1]
            parsed_conll = FileFormatParser.parse_t15conll_file(file_content.decode("utf-8"))
            for tweet in parsed_conll:
                tweet["label"] = await self.__process_label_references(tweet["label"])
            json_processed = reduce(lambda old, new: f"{old}\n{new}",
                                    map(lambda cnl: self.__dict_to_json(cnl), parsed_conll))
            if file_name == "dev":
                file_name = "val"
            await file_writer(f"{self.output_path}/text_preprocessed/t15_{file_name}.txt",
                              json_processed.encode("utf-8"))

    async def __load_twitter_image_dataset(self, save_file_consumer: Callable[[asyncio.Queue], Coroutine]):
        image_path = f"{self.input_path}/twitter2015_images"
        queue = asyncio.Queue(maxsize=2 ** 10)
        await asyncio.gather(filesystem.load_directory_contents(image_path, queue),
                             self.__process_twitter15_image_files(queue, save_file_consumer))

    async def __process_label_references(self, labels: list[str]) -> list[str]:
        mapped_labels = self.label_processor.process_data(labels).result(2000)
        return mapped_labels


    async def __process_twitter15_image_files(self, queue: asyncio.Queue,
                                              save_file_consumer: Callable[[asyncio.Queue], Coroutine]):
        processed_images_que = asyncio.Queue(maxsize=2 ** 10)
        asyncio.create_task(save_file_consumer(processed_images_que))
        while True:
            file_wrapper = await queue.get()
            if file_wrapper is None:
                break
            file_name = file_wrapper[0].split(".")[0]
            file_content = file_wrapper[1]
            result = self.image_processor.process_data(file_content)
            resized_image = result.result(2500)
            await processed_images_que.put((f"{self.output_path}/image_preprocessed/{file_name}.jpeg", resized_image))
        await processed_images_que.put(None)

    @staticmethod
    def __dict_to_json(dic: dict[str, list]):
        return json.dumps(dic)
