import asyncio
import time

import torch
from huggingface_hub import login

import data.dataset_preprocessor as data_preprocessor
from data.data_processors import process_twitter2017_text, process_twitter2017_image
from data.dataset import load_twitter_dataset
from model.configuration.quantization import create_default_quantization_config, create_parameter_efficient_model
from model.model_factory import (create_model, create_model_for_lm, create_roberta_base, create_vit)
from model.multimodal.text_image_model import CombinedModel
from model.util import load_and_filter_state_dict, create_message
from model.visual.convolutional_net import ConvNet
from model.visual.vit_wrapper import  ViT
from security.token_manager import TokenManager
from train import train
from train.train import training_loop_combined


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
    train.inference_loop_combined_model(combined, data['test'], tokenizer)


async def preprocess_twitter():
    print("Loading dataset, image and text")
    start = time.time()
    await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")


async def create_roberta_multimodal():
    # print("Loading dataset")
    data, labels = await load_twitter_dataset()
    # print("Dataset loaded")
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    model, tokenizer = create_roberta_base()
    cnn = ConvNet()
    combined = CombinedModel(cnn, model, len(labels.keys()))
    training_loop_combined(combined, data["train"], data["val"], tokenizer)


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
    combined = create_parameter_efficient_model(CombinedModel(cnn, model, len(labels.keys())))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], tokenizer, list(labels.values()))
    combined.merge_and_unload()
    print("Saving model")
    MODEL_OUT_PATH = "./combined_model.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")
    # text_input = tokenizer(" ".join(text[1][0][0][0]), return_tensors="pt")


async def llama_vit_multimodal():
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    data, labels = await load_twitter_dataset()
    print("Dataset loaded")

    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    vit_model, vit_processor = create_vit()
    vit = ViT(vit_model, vit_processor)
    # cnn(torch.rand(3, 256, 256))
    print("Creating model")
    model, tokenizer = create_model(model_name, create_default_quantization_config())
    combined = create_parameter_efficient_model(CombinedModel(vit, model, len(labels.keys())))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], tokenizer)
    combined.merge_and_unload()
    print("Saving model")
    MODEL_OUT_PATH = "./combined_model.pth"
    torch.save(combined.state_dict(), MODEL_OUT_PATH)
    print("Leaving")


async def ner_prompt():
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model, tokenizer = create_model_for_lm(model_name, create_default_quantization_config())
    data, labels = await load_twitter_dataset()
    system_text = f"Perform named entity recognition using only these entities: {labels}. Answer in format 'token':'label' on new line for each token."
    msg = create_message(system_text, " ".join(data['test'][10][0]))

    tokenizer_out = tokenizer(msg, return_tensors="pt")
    outputs = model.generate(input_ids=tokenizer_out["input_ids"], max_new_tokens=100,
                             attention_mask=tokenizer_out["attention_mask"])
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)


def create_random_tensors(dim: tuple) -> torch.Tensor:
    return torch.randint(low=1, high=9, size=dim, dtype=torch.int)


async def run_vit():
    model, processor = create_vit()
    r_image = torch.rand((3, 256, 256), dtype=torch.float32)
    inputs = processor(r_image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state
    print(embedding)


if __name__ == "__main__":
    '''
    y_pred = []
    y_true = []
    for i in range(100):
        l = randint(9, 20)
        y_pred.append(create_random_tensors((l, 9)))
        y_true.append(create_random_tensors((l,)))
    y_pred = torch.from_numpy(np.array(y_pred, dtype=np.int8))
    y_true = torch.from_numpy(np.array(y_true))
    m = Metrics(torch.argmax(y_pred,dim = -1), torch.from_numpy(y_true))
    acc, f1 =  m.f1(), m.acc()
    print(f"acc: {acc}, f1: {f1}")
    # asyncio.run(create_roberta_multimodal())
    '''
    asyncio.run(llama_vit_multimodal())
