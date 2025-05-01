import asyncio
import time

from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor


async def preprocess_twitter17():
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = Twitter2017Preprocessor()
    await preprocessor.load_and_transform_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


async def preprocess_twitter15():
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = Twitter2015Preprocessor()
    await preprocessor.load_and_transform_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


if __name__ == "__main__":
    asyncio.run(preprocess_twitter15())
    asyncio.run(preprocess_twitter17())