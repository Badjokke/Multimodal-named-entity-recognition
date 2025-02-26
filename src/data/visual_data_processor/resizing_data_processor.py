from concurrent.futures import ThreadPoolExecutor, Future
from typing import Union

import cv2 as opencv
import numpy as np
from torch import Tensor

from data.data_processor import DataProcessor


class ResizingDataProcessor(DataProcessor):
    def __init__(self, dim=(224, 224)):
        super().__init__()
        self.io_pool_exec = ThreadPoolExecutor(max_workers=5)
        self.dim = dim

    def process_data(self, data: Union[bytes, str, Tensor]) -> Future[bytes]:
        return self.io_pool_exec.submit(self.__resize_image, data, self.dim)

    @staticmethod
    def __resize_image(image_binary: bytes, dim: tuple[int, int]) -> bytes:
        img_decoded = opencv.imdecode(np.frombuffer(image_binary, dtype=np.uint8), opencv.IMREAD_COLOR)
        buffer = opencv.resize(img_decoded, dim)
        success, buffer = opencv.imencode(".jpg", buffer)
        if not success:
            raise Exception("Better exception in future.")
        return buffer.tobytes()
