import asyncio
import os
from typing import Callable, Coroutine, AsyncGenerator

import aiofiles
MAX_DEPTH = 3
async def load_file(path: str) -> bytes:
    async with aiofiles.open(path, 'rb') as f:
        return await f.read()


async def file_lines_generator(path: str) -> AsyncGenerator[str, None]:
    async with aiofiles.open(path, 'r') as f:
        async for line in f:
            yield line
    yield "EOF"


async def save_file_consumer(queue: asyncio.Queue[tuple[str, bytes]]):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        path = item[0]
        content = item[1]
        await save_file(path, content)
        queue.task_done()


async def save_file(path: str, content: bytes):
    assert path is not None
    dir_parent = "/".join(path.split("/")[:-1])
    if not os.path.exists(dir_parent):
        os.mkdir(dir_parent)
    async with aiofiles.open(path, 'wb') as f:
        await f.write(content)


async def _load_directory_contents(path, queue: asyncio.Queue, file_parser: Callable[[str], Coroutine],include_parent_dir, level=0):
    children = os.listdir(path)
    parent = path.split(os.path.sep)[-1]
    for i in range(0, len(children)):
        p = os.path.join(path, children[i])
        if not os.path.isdir(p):
            await queue.put((children[i] if not include_parent_dir else f"{parent}/{children[i]}", await file_parser(p)))
            continue
        if level < MAX_DEPTH:
            await _load_directory_contents(p, queue, file_parser, level=level + 1, include_parent_dir=include_parent_dir)
    if level == 0:
        await queue.put(None)

async def load_directory_contents(path: str, queue: asyncio.Queue, include_parent_dir=False):
    await _load_directory_contents(path, queue, load_file, include_parent_dir)


async def load_directory_contents_generator(path: str, queue: asyncio.Queue):
    children = os.listdir(path)
    for i in range(0, len(children)):
        gen = file_lines_generator(os.path.join(path, children[i]))
        while True:
            elem = await gen.__anext__()
            if elem == "EOF":
                break
            await queue.put(("line", elem))
        await queue.put((children[i], "EOF"))
    await queue.put(None)
