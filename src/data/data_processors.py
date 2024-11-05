import json
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor

import cv2 as opencv
import numpy as np
from nltk.stem import PorterStemmer
from torch import from_numpy

io_pool_exec = ThreadPoolExecutor(max_workers=10)
ps = PorterStemmer()


def process_twitter2017_text(sentence: str) -> Future[tuple[list[str], list[str], list[str]]]:
    return io_pool_exec.submit(_process_json_text, sentence)


def process_twitter2017_image(image_binary: bytes, extension: str) -> Future[bytes]:
    return io_pool_exec.submit(_process_image, image_binary, extension)


def _process_image(image_binary: bytes, extension: str) -> bytes:
    if not extension.startswith('.'):
        extension = '.' + extension
    img_decoded = opencv.imdecode(np.frombuffer(image_binary, dtype=np.uint8), opencv.IMREAD_COLOR)
    buffer = opencv.resize(img_decoded, (256, 256))
    success, buffer = opencv.imencode(extension, buffer)
    if not success:
        raise Exception("Better exception in future.")
    return buffer.tobytes()


def _process_json_text(text: str) -> tuple[list[str], list[str], list[str]]:
    json_value = json.loads(text)
    text = json_value.get('text')
    labels = json_value.get('label')
    related_images = json_value.get('images')
    return _stem_text(text), related_images, labels


def _stem_text(text: list[str]) -> list[str]:
    stemmed_text = []
    for i in range(len(text)):
        stemmed_text.append(ps.stem(text[i]))
    return stemmed_text


# TODO fix preprocessed json
def _parse_twitter_text(content: bytes):
    lines = content.decode('utf-8').split('\n')
    list = []
    for line in lines:
        jsonl = json.loads(line)
        list.append((jsonl["text"], jsonl["images"], jsonl["label"]))
    return list

def parse_twitter_text(content:bytes):
    return io_pool_exec.submit(_parse_twitter_text, content)


def image_to_tensor(image: bytes):
    return io_pool_exec.submit(_image_to_tensor, image)

# TODO returns nil tensor
def _image_to_tensor(image: bytes):
    img_decoded = opencv.imdecode(np.frombuffer(image, dtype=np.uint8), opencv.IMREAD_COLOR)
    img_decoded = opencv.cvtColor(img_decoded, opencv.COLOR_BGR2RGB)
    tensor = from_numpy(img_decoded).permute(2, 0, 1)
    return tensor
