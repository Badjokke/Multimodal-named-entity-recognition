import asyncio
import os
from typing import Callable, Coroutine

import aiofiles


async def load_file(path: str):
    async with aiofiles.open(path, 'rb') as f:
        return await f.read()


async def file_lines_generator(path: str):
    async with aiofiles.open(path, 'r') as f:
        async for line in f:
            yield line
    yield "EOF"
    return


async def save_file(path: str, content: bytes):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(content)


async def __load_directory_contents(path, queue: asyncio.Queue, file_parser: Callable[[str], Coroutine]):
    children = os.listdir(path)
    for i in range(0, len(children)):
        await queue.put((children[i], await file_parser(os.path.join(path, children[i]))))
    await queue.put(None)


async def load_directory_contents(path: str, queue: asyncio.Queue):
    await __load_directory_contents(path, queue, load_file)

# todo read all files at once
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
