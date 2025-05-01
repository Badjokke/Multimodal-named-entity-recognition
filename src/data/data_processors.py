import json
from concurrent.futures import Future, ThreadPoolExecutor

import cv2 as opencv
import numpy as np
from torch import from_numpy

io_pool_exec = ThreadPoolExecutor(max_workers=10)

import base64

def parse_twitter_text(content: bytes) -> Future[list[dict[str, str]]]:
    return io_pool_exec.submit(_parse_twitter_text, content)


def image_to_tensor(image: bytes):
    return io_pool_exec.submit(_image_to_tensor, image)


def _image_to_tensor(image: bytes):
    img_decoded = opencv.imdecode(np.frombuffer(image, dtype=np.uint8), opencv.IMREAD_COLOR)
    img_decoded = opencv.cvtColor(img_decoded, opencv.COLOR_BGR2RGB)
    tensor = from_numpy(img_decoded).permute(2, 0, 1)
    return tensor / 255

def image_to_base64(image: bytes) -> Future[str]:
    return io_pool_exec.submit(_image_to_base64, image)

def _image_to_base64(image: bytes) -> str:
    return base64.b64encode(image).decode("utf-8")


def _parse_twitter_text(content: bytes) -> list[dict[str, str]]:
    lines = content.decode('utf-8').split('\n')
    tmp = []
    for line in lines:
        if line == '\n' or line == "":
            continue
        jsonl = json.loads(line)
        tmp.append(jsonl)
    return tmp