import asyncio
import time

import torch
from huggingface_hub import login

from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor
from model.configuration.quantization import create_default_quantization_config
from model.model_factory import (create_llama_model, create_vit, create_lstm, create_bert_large)
from model.multimodal.text_image_model import CombinedModel
from model.util import load_and_filter_state_dict
from model.visual.convolutional_net import ConvNet
from model.visual.vit_wrapper import ViT
from security.token_manager import TokenManager
from train import train
from typing import Callable, Coroutine
from data.dataset_analyzer import DatasetAnalyzer
from metrics.plot_builder import PlotBuilder

async def inference(model_path):
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    print("Dataset loaded")
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    cnn = ConvNet()
    print("Creating model")
    model, tokenizer = create_llama_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels.keys()))

    state_dict = load_and_filter_state_dict(model_path)
    combined.load_state_dict(state_dict)
    train.inference_loop_combined_model(combined, data['test'], tokenizer)


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


def merge_lora_layers_with_text_model(combined_model: CombinedModel) -> torch.nn.Module:
    return combined_model.text_model.merge_and_unload()


async def create_vit_lstm_model():
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    lstm = create_lstm(len(vocabulary.keys()))
    vit, processor = create_vit()
    vit = ViT(vit, processor)
    combined = CombinedModel(vit, lstm, len(labels.keys()))
    combined = train.training_loop_combined(combined, data["train"], data["val"], data["test"], None, vocabulary,
                                            class_occurrences)


async def llama_vit_multimodal():
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    print("Dataset loaded")

    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    vit_model, vit_processor = create_vit()
    vit = ViT(vit_model, vit_processor)
    # cnn(torch.rand(3, 256, 256))
    print("Creating vit llama model")
    model, tokenizer = create_llama_model(model_name, create_default_quantization_config())
    combined = CombinedModel(vit, create_bert_large(), len(labels.keys()))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], data["test"], tokenizer,
                                            class_occurrences, labels,
                                            epochs=10)
    # combined.text_model = merge_lora_layers_with_text_model(combined)
    print("Saving model")
    combined.save_pretrained("peft_finetuned_llama.pth")
    MODEL_OUT_PATH = "./combined_model_roberta_vit.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")

async def analyze_dataset(dataset_loader:Callable[[], Coroutine]):
    data, labels, class_occurrences, vocabulary = await dataset_loader()
    analyzer = DatasetAnalyzer(data, class_occurrences, labels, vocabulary)
    class_count = analyzer.get_dataset_label_count()
    unique_token_count = analyzer.get_unique_token_count()
    print(f"Class count: {class_count}")
    print(f"Unique tokens: {unique_token_count}")
    bar = PlotBuilder.build_cake_plot(list(class_count.values()), list(class_count.keys()), plot_label="T17 distribution")
    bar.plot()
    print("==train==")
    train_dataset = analyzer.get_train_subset_stats()
    print(train_dataset)
    bar = PlotBuilder.build_cake_plot(list(train_dataset["hist"].values()), list(train_dataset["hist"].keys()), plot_label="T17 train set")
    bar.plot()
    print("==val==")
    validation = analyzer.get_validation_subset_stats()

    bar = PlotBuilder.build_cake_plot(list(validation["hist"].values()), list(validation["hist"].keys()), plot_label="T17 validation set")
    bar.plot()
    print("==test==")
    test_dataset = analyzer.get_test_subset_stats()
    bar = PlotBuilder.build_cake_plot(list(test_dataset["hist"].values()), list(test_dataset["hist"].keys()), plot_label="T17 test set")
    bar.plot()


if __name__ == "__main__":
    #asyncio.run(preprocess_twitter15())
    #asyncio.run(llama_vit_multimodal())
    #asyncio.run(preprocess_twitter15())
    t17_loader = JsonlDatasetLoader(lightweight=True)
    asyncio.run(analyze_dataset(t17_loader.load_dataset))
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
