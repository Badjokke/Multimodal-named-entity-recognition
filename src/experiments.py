import asyncio
import time
from typing import Callable, Coroutine

from huggingface_hub import login

from data.dataset_analyzer import DatasetAnalyzer
from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor
from metrics.plot_builder import PlotBuilder
from model.model_factory import ModelFactory
from model.util import plot_model_training
from security.token_manager import TokenManager
from train import train


async def preprocess_twitter():
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


async def analyze_dataset(dataset_loader: Callable[[], Coroutine]):
    data, labels, class_occurrences, vocabulary = await dataset_loader()
    analyzer = DatasetAnalyzer(data, class_occurrences, labels, vocabulary)
    class_count = analyzer.get_dataset_label_count()
    unique_token_count = analyzer.get_unique_token_count()
    print(f"Class count: {class_count}")
    print(f"Unique tokens: {unique_token_count}")
    bar = PlotBuilder.build_cake_plot(list(class_count.values()), list(class_count.keys()),
                                      plot_label="T17 distribution")
    bar.plot()
    print("==train==")
    train_dataset = analyzer.get_train_subset_stats()
    print(train_dataset)
    bar = PlotBuilder.build_cake_plot(list(train_dataset["hist"].values()), list(train_dataset["hist"].keys()),
                                      plot_label="T17 train set")
    bar.plot()
    print("==val==")
    validation = analyzer.get_validation_subset_stats()

    bar = PlotBuilder.build_cake_plot(list(validation["hist"].values()), list(validation["hist"].keys()),
                                      plot_label="T17 validation set")
    bar.plot()
    print("==test==")
    test_dataset = analyzer.get_test_subset_stats()
    bar = PlotBuilder.build_cake_plot(list(test_dataset["hist"].values()), list(test_dataset["hist"].keys()),
                                      plot_label="T17 test set")
    bar.plot()


async def multimodal_pipeline(model_save_directory: str):
    print("Running multimodal pipeline")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()

    bert_vit, tokenizer = ModelFactory.create_bert_vit_attention_classifier(len(labels.keys()))
    print("Training bert+vit")
    combined, results = train.multimodal_training(bert_vit, data['train'], data["val"], data["test"], tokenizer,
                                                  class_occurrences, labels, epochs=10, patience=2)
    plot_model_training(results, f"{model_save_directory}/fig/plot.png")
    """
    torch.save(combined.state_dict(), model_save_directory + "/bert_vit_cross_attention.pth")
    llama_vit, tokenizer = ModelFactory.create_llama_vit_attention_classifier(len(labels.keys()))
    # combined.text_model = merge_lora_layers_with_text_model(combined)
    print("Saving model")
    combined.save_pretrained("peft_finetuned_llama.pth")
    MODEL_OUT_PATH = "./combined_model_roberta_vit.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")
    """


if __name__ == "__main__":
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    # asyncio.run(preprocess_twitter15())
    # asyncio.run(llama_vit_multimodal())
    # asyncio.run(preprocess_twitter15())
    # t17_loader = JsonlDatasetLoader(lightweight=True)
    asyncio.run(multimodal_pipeline("../models/bert/t17"))
    """
    random_input = torch.LongTensor([1, 2, 3, 4, 5, 6, 7])
    lstm = create_lstm(24)
    features = lstm(random_input)
    print(features)
    """
    """
    # asyncio.run(create_roberta_multimodal())

    # asyncio.run(llama_vit_multimodal())
    """
