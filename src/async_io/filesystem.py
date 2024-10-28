import asyncio
import os
import aiofiles


async def load_file(path: str):
    async with aiofiles.open(path, 'rb') as f:
        return await f.read()


async def save_file(path: str, content: bytes):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(content)


async def load_directory_contents(path: str, queue: asyncio.Queue):
    children = os.listdir(path)
    for i in range(0, len(children)):
        await queue.put((children[i], await load_file(os.path.join(path, children[i]))))
    await queue.put(None)
