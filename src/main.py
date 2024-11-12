import asyncio

import torch
from huggingface_hub import login

from data.dataset import load_twitter_dataset
from model.configuration.quantization import create_default_quantization_config
from model.model_factory import (create_model)
from model.multimodal.text_image_model import CombinedModel
from model.visual.convolutional_net import ConvNet
from security.token_manager import TokenManager
from train import train


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
    # image_tensor =image[1]["0_0.jpg"][None,:,:,:] / 255  # Example image batch (1, 3, 64, 64)
    cnn = ConvNet()
    # cnn(image_tensor)
    print("Creating model")
    model, tokenizer = create_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels))
    print("Training combined model")
    combined = train.training_loop(combined, data['train'], tokenizer)
    print("Saving model")
    MODEL_OUT_PATH = "./combined_model.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")
    # text_input = tokenizer(" ".join(text[1][0][0][0]), return_tensors="pt")


if __name__ == "__main__":
    asyncio.run(main())
