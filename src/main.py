import asyncio
import time

import torch
from huggingface_hub import login

from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor
from model.configuration.quantization import create_default_quantization_config, create_parameter_efficient_model
from model.model_factory import (create_llama_model, create_vit, create_lstm, create_bert_large, create_convolutional_net)
from model.multimodal.CrossAttentionMultimodalModel import CrossAttentionModel
from model.util import load_and_filter_state_dict
from model.visual.AlexNetCNN import ConvNet
from model.visual.ViTWrapper import ViT
from security.token_manager import TokenManager
from data.text_data_processor.stemming_json_data_processor import StemmingTextDataProcessor
from train import train


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
    combined = CrossAttentionModel(cnn, model, len(labels.keys()))

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


def merge_lora_layers_with_text_model(combined_model: CrossAttentionModel) -> torch.nn.Module:
    return combined_model.text_model.merge_and_unload()


async def create_vit_lstm_model():
    t17_loader = JsonlDatasetLoader(text_processors=[StemmingTextDataProcessor()])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    lstm = create_lstm(vocabulary)
    vit, processor = create_vit()
    vit = ViT(vit, processor)
    combined = CrossAttentionModel(vit, lstm, len(labels.keys()))
    combined = train.training_loop_combined(combined, data['train'], data["val"], data["test"], None,
                                            class_occurrences, labels,
                                            epochs=20)
    # combined.text_model = merge_lora_layers_with_text_model(combined)
    print("Saving model lstm_vit_crossattention")
    MODEL_OUT_PATH = "../models/lstm/t17/lstm_vit_crossattention.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)


async def llama_vit_multimodal():
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    token_manager = TokenManager()

    login(token_manager.get_access_token())

    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    print("Dataset loaded")

    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    login(token_manager.get_access_token())
    vit_model, vit_processor = create_vit()
    vit = ViT(vit_model, vit_processor)
    # cnn(torch.rand(3, 256, 256))
    #cnn = create_convolutional_net()
    print("Creating vit llama model")
    model, tokenizer = create_bert_large()
    #model,tokenizer = create_llama_model(model_name, create_default_quantization_config())
    #model = create_lstm(len(vocabulary.keys()), True)
    combined = CrossAttentionModel(vit, model, len(labels.keys()))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], data["test"], tokenizer,
                                            class_occurrences, labels,
                                            epochs=15)
    # combined.text_model = merge_lora_layers_with_text_model(combined)
    print("Saving model")
    MODEL_OUT_PATH = "../models/bert/t32/cross_attention_bert_vit.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")


if __name__ == "__main__":
    asyncio.run(create_vit_lstm_model())
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
