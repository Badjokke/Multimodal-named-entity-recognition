from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
import cv2 as opencv
import numpy as np
io_pool_exec = ThreadPoolExecutor(max_workers=4)


def process_sentence(sentence: str) -> Future[str]:
    return io_pool_exec.submit(__process_text, sentence)



def process_image(image_binary: bytes, extension: str) -> Future[bytes]:
    return io_pool_exec.submit(__process_image, image_binary, extension)


def __process_image(image_binary: bytes, extension: str) -> bytes:
    if not extension.startswith('.'):
        extension = '.' + extension
    img_decoded = opencv.imdecode(np.frombuffer(image_binary, dtype=np.uint8), opencv.IMREAD_COLOR)
    buffer = opencv.resize(img_decoded, (256, 256))
    success, buffer = opencv.imencode(extension, buffer)
    if not success:
        raise Exception("Better exception in future.")
    return buffer.tobytes()

def __process_text(text: str) -> str:
    print(text)
    return text
