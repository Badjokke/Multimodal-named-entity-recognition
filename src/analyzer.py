import argparse
import asyncio
from typing import Callable, Coroutine

from data.dataset_analyzer import DatasetAnalyzer
from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from metrics.plot_builder import PlotBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Provide a dataset root path for analysis")
    parser.add_argument(
        "--dataset",
        help="Dataset root. Directory must have direct children text_preprocessed and image_preprocessed!", type=str
    )
    #last minute vohejbak
    parser.add_argument(
        "--SOA",
        help="If SOA dataset is being analyzed. Different loader is used for that dataset.", type=str
    )
    return parser.parse_args()


async def analyze_dataset(dataset_loader: Callable[[], Coroutine]):
    data, labels, class_occurrences, vocabulary = await dataset_loader()
    analyzer = DatasetAnalyzer(data, class_occurrences, labels, vocabulary)
    class_count = analyzer.get_dataset_label_count()
    unique_token_count = analyzer.get_unique_token_count()
    print(f"Class count: {class_count}")
    print(f"Unique tokens: {unique_token_count}")
    bar = PlotBuilder.build_cake_plot(list(class_count.values()), list(class_count.keys()),
                                      plot_label="Dataset distribution")
    bar.plot()
    print("==train==")
    train_dataset = analyzer.get_train_subset_stats()
    print(train_dataset)
    bar = PlotBuilder.build_cake_plot(list(train_dataset["hist"].values()), list(train_dataset["hist"].keys()),
                                      plot_label="Train set distribution")
    bar.plot()
    print("==val==")
    validation = analyzer.get_validation_subset_stats()
    print(validation)
    bar = PlotBuilder.build_cake_plot(list(validation["hist"].values()), list(validation["hist"].keys()),
                                      plot_label="Validation set distribution")
    bar.plot()
    print("==test==")
    test_dataset = analyzer.get_test_subset_stats()
    print(test_dataset)
    bar = PlotBuilder.build_cake_plot(list(test_dataset["hist"].values()), list(test_dataset["hist"].keys()),
                                      plot_label="Test set distribution")
    bar.plot()


if __name__ == "__main__":
    args = parse_args()
    dataset = parse_args().dataset
    loader = JsonlDatasetLoader(input_path=dataset, lightweight=True) if not args.SOA or args.SOA.upper() == "FALSE" else JsonlDatasetLoader(input_path=dataset, lightweight=True, include_parent_dir=True, custom_split=True)
    asyncio.run(analyze_dataset(loader.load_dataset))
