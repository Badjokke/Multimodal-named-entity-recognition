import asyncio
import time

import torch
from huggingface_hub import login

from data.dataset import load_twitter_dataset, dataset_text_only
from data.dataset_preprocessor import TwitterPreprocessor
from data.text_data_processor.stemming_json_data_processor import StemmingTextDataProcessor
from model.configuration.quantization import create_default_quantization_config, create_parameter_efficient_model
from model.model_factory import (create_llama_model, create_vit, create_lstm, create_bert_large, create_model_for_lm)
from model.multimodal.text_image_model import CombinedModel
from model.language.llama import LlamaLM
from model.util import load_and_filter_state_dict,save_model, create_message, freeze_model, load_lora_model, save_lora_model
from model.visual.convolutional_net import ConvNet
from model.visual.vit_wrapper import ViT
from security.token_manager import TokenManager
from train import train
from train.train import training_loop_combined, training_loop_text_only


async def inference(model_path):
    print("Loading dataset")
    data, labels, class_occurrences, vocabulary = await load_twitter_dataset()
    print("Dataset loaded")
    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    # cnn(torch.rand(3, 256, 256))
    print("Creating model")
    model, processor = create_vit()
    vit = ViT(model, processor)
    model, tokenizer = create_bert_large()
    combined = CombinedModel(vit, model, len(labels.keys()))

    state_dict = load_and_filter_state_dict(model_path)
    combined.load_state_dict(state_dict)
    loss, metrics = train.validate_after_epoch(combined,tokenizer,None, data['test'])
    print(f"loss: {loss}")
    print(f"metrics: {metrics}")

async def preprocess_twitter():
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = TwitterPreprocessor()
    await preprocessor.load_twitter_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")

async def create_vit_lstm_model():
    data, labels, class_occurrences, vocabulary = await load_twitter_dataset(text_processors=[StemmingTextDataProcessor()])
    lstm = create_lstm(len(vocabulary))
    vit, processor = create_vit()
    vit = ViT(vit, processor)
    combined = CombinedModel(vit, lstm, len(labels.keys()))
    combined = training_loop_combined(combined, data["train"], data["val"], data["test"], class_occurrences, epochs=20)
    combined.save_pretrained("vit_lstm.pth")



async def llama_vit_multimodal():
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    data, labels, class_occurrences, vocabulary = await load_twitter_dataset()
    print("Dataset loaded")

    token_manager = TokenManager()
    login(token_manager.get_access_token())

    vit_model, vit_processor = create_vit()
    vit = ViT(vit_model, vit_processor)
    print("Creating vit llama model")
    #model, tokenizer = create_llama_model(model_name, create_default_quantization_config())
    model, tokenizer = create_bert_large()
    combined = CombinedModel(vit,(model),len(labels.keys()))

    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], data["test"], tokenizer,class_occurrences,labels, epochs=15)
    torch.save(combined.state_dict(), "models/bert_vit_crf_bilstm_rarevents.pth")
    #save_model(combined, "models/llama_vit_raw.pth")
    #combined.text_model = combined.text_model.merge_and_unload()
    print("Saving model")
    #save_lora_model(combined,"peft_models")
    print("Leaving")



if __name__ == "__main__":
    asyncio.run(llama_vit_multimodal())
    #asyncio.run(llama_vit_multimodal())
    """
    model, tokenizer = create_model_for_lm("meta-llama/Llama-3.1-8B", create_default_quantization_config())
    extracted = []
    llama = LlamaLM(model,tokenizer)
    sample_text = " ".join(["DCC", "caption", ":", "Gahyeon", "was", "shy", "because", "of", "Bora", "s", "hand", "for", "a", "moment",
     "!", "lol"])
    entities = llama.extract_entities(sample_text)
    extracted.append(entities)

    sample_text = " ".join(["Dressed", "as", "Shin", "Hayata", ".", "Happy", "Halloween", "!", "!", "!", "!", "#", "ultraman", "#", "halloween2022"])
    entities = llama.extract_entities(sample_text)
    extracted.append(entities)
    print("\nDetected entities:")
    for i in range(len(extracted)):
        for entity, type_ in extracted[i]:
            print(f"- {entity}: {type_}")
    """