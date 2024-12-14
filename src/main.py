import time
from random import randint

import torch
from huggingface_hub import login
from metrics.plot_builder import PlotBuilder
import data.dataset_preprocessor as data_preprocessor
from data.data_processors import process_twitter2017_text, process_twitter2017_image
from data.dataset import load_twitter_dataset
from metrics.metrics import Metrics
from metrics.plots import SimplePlot
from model.configuration.quantization import create_default_quantization_config, create_parameter_efficient_model
from model.model_factory import (create_model, create_model_for_lm, create_roberta_base, create_vit)
from model.multimodal.text_image_model import CombinedModel
from model.util import load_and_filter_state_dict, create_message
from model.visual.convolutional_net import ConvNet
from model.visual.vit_wrapper import ViT
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


def merge_lora_layers_with_text_model(combined_model: CombinedModel) -> torch.nn.Module:
    return combined_model.text_model.merge_and_unload()


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
    combined = CombinedModel(vit, create_parameter_efficient_model(model), len(labels.keys()))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], tokenizer, epochs=1)
    combined.text_model = merge_lora_layers_with_text_model(combined)
    print("Saving model")
    MODEL_OUT_PATH = "./combined_model_llama_vit.pth"
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
    return torch.randint(low=0, high=3, size=dim, dtype=torch.int)


async def run_vit():
    model, processor = create_vit()
    r_image = torch.rand((3, 256, 256), dtype=torch.float32)
    inputs = processor(r_image, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state
    print(embedding)


if __name__ == "__main__":
    x = [[1,2,3],[3,2,1]]
    y = [[1,2,3],[1,2,3]]
    simple_plot = PlotBuilder.build_simple_plot(x,y, colors=["gold","red"])
    simple_plot.plot()

    '''
    y_pred = []
    y_true = []
    for i in range(20):
        l = randint(3, 20)
        y_pred.append(torch.argmax(create_random_tensors((l, 3)), dim=-1))
        y_true.append(create_random_tensors((l,)))
    m = Metrics(y_pred, y_true, 3, {0: "dog", 1: "cat", 2: "horse"})
    matrix = m.confusion_matrix()
    matrix.print_matrix()
    m.f1(matrix,0)
    print(m.macro_f1(matrix))
    # asyncio.run(create_roberta_multimodal())

    # asyncio.run(llama_vit_multimodal())
    '''