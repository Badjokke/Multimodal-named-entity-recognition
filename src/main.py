import asyncio
import time

import torch
from huggingface_hub import login

import data.dataset_preprocessor as data_preprocessor
from data.data_processors import process_twitter2017_text, process_twitter2017_image
from data.dataset import load_twitter_dataset
from model.configuration.quantization import create_default_quantization_config
from model.model_factory import (create_model)
from model.multimodal.text_image_model import CombinedModel
from model.util import load_and_filter_state_dict
from model.visual.convolutional_net import ConvNet
from security.token_manager import TokenManager
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


async def inference(model_path):
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    data, labels = await load_twitter_dataset()
    print("Dataset loaded")
    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    cnn = ConvNet()
    # cnn(torch.rand(3, 256, 256))
    print("Creating model")
    model, tokenizer = create_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels.keys()))

    state_dict = load_and_filter_state_dict(model_path)
    combined.load_state_dict(state_dict)
    train.inference_loop(combined, data['test'], tokenizer)


async def preprocess_twitter():
    print("Loading dataset, image and text")
    start = time.time()
    await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


async def main():
    '''
    print("Loading dataset, image and text")
    start = time.time()
    await dataset_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")
    exit(1)
    '''
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    data, labels = await load_twitter_dataset()
    print("Dataset loaded")

    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    cnn = ConvNet()
    # cnn(torch.rand(3, 256, 256))
    print("Creating model")
    model, tokenizer = create_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels.keys()))
    print("Training combined model")
    combined = train.training_loop(combined, data['train'], tokenizer)
    print("Saving model")
    MODEL_OUT_PATH = "./combined_model.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")
    # text_input = tokenizer(" ".join(text[1][0][0][0]), return_tensors="pt")


if __name__ == "__main__":
    asyncio.run(preprocess_twitter())
