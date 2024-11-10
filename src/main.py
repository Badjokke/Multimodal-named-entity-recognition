from huggingface_hub import login
import asyncio
from model.train import train
from model.custom_models.convolutional_net import ConvNet
from model.custom_models.text_image_model import CombinedModel
#from data.dataset import (load_hf_dataset)
from model.quantization import create_default_quantization_config
from model.model_factory import (create_model, create_multimodal_model)
#from model.train.trainer import (train_transformer_model)
from security.token_manager import get_access_token
from data.dataset import load_twitter_dataset
import torch
"""
def ner_classify():
    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = create_model(model_name, None)
    ds = preprocess_dataset_class(load_hf_dataset(), tokenizer)

    train_transformer_model(model, tokenizer, ds)
"""

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
    data, labels = await load_twitter_dataset()

    #cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    login(get_access_token())
    #image_tensor =image[1]["0_0.jpg"][None,:,:,:] / 255  # Example image batch (1, 3, 64, 64)
    cnn = ConvNet()
    #cnn(image_tensor)
    model, tokenizer = create_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels))
    combined = train.training_loop(combined, data['train'], tokenizer)

    MODEL_OUT_PATH = "./combined_model.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)


    #text_input = tokenizer(" ".join(text[1][0][0][0]), return_tensors="pt")

    #output = combined(image_tensor, text_input)
    print("out")
    #print(output)
    print("out")

    # ner_classify()


if __name__ == "__main__":
    asyncio.run(main())
