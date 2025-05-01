import asyncio
import time
import argparse
from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Optional T17 and T15 directory path arguments.")

    parser.add_argument('--t17', type=str, default=None,
                        help='Path to the T17 directory (optional)')
    parser.add_argument('--t15', type=str, default=None,
                        help='Path to the T15 directory (optional)')

    parser.add_argument('--t17_out', type=str, default=None,
                        help='Path to the T15 output directory (optional)')
    parser.add_argument('--t17_out', type=str, default=None,
                        help='Path to the T17 output directory (optional)')
    return parser.parse_args()

async def preprocess_twitter17(twitter17_path, dataset_out):
    assert dataset_out is not None, "T17 dataset output is not provided"
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = Twitter2017Preprocessor(input_path=twitter17_path, output_path=dataset_out)
    await preprocessor.load_and_transform_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


async def preprocess_twitter15(twitter15_path, dataset_out):
    assert dataset_out is not None, "T15 dataset output is not provided"
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = Twitter2015Preprocessor(input_path=twitter15_path, output_path=dataset_out)
    await preprocessor.load_and_transform_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


if __name__ == "__main__":
    args = parse_args()
    if args.t15 is not None:
        asyncio.run(preprocess_twitter15(args.t15, args.t15_out))
    if args.t17 is not None:
        asyncio.run(preprocess_twitter17(args.t17, args.t17_out))