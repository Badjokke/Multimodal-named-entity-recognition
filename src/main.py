import asyncio
import time
from random import randint

import torch
from huggingface_hub import login

from data.dataset import load_twitter_dataset, dataset_text_only
from data.dataset_preprocessor import TwitterPreprocessor
from metrics.metrics import Metrics
from metrics.plot_builder import PlotBuilder
from metrics.plots import SimplePlot
from model.configuration.quantization import create_default_quantization_config, create_parameter_efficient_model
from model.model_factory import (create_llama_model, create_model_for_lm, create_roberta_base, create_vit, create_mistral, create_token_classification_llama)
from model.multimodal.text_image_model import CombinedModel
from model.util import load_and_filter_state_dict, create_message
from model.visual.convolutional_net import ConvNet
from model.visual.vit_wrapper import ViT
from security.token_manager import TokenManager
from train import train
from train.train import training_loop_combined, training_loop_text_only


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
    model, tokenizer = create_llama_model(model_name, create_default_quantization_config())
    combined = CombinedModel(cnn, model, len(labels.keys()))

    state_dict = load_and_filter_state_dict(model_path)
    combined.load_state_dict(state_dict)
    train.inference_loop_combined_model(combined, data['test'], tokenizer)


async def preprocess_twitter():
    print("Loading dataset, image and text")
    start = time.time()
    preprocessor = TwitterPreprocessor()
    await preprocessor.load_twitter_dataset()
    # await data_preprocessor.load_twitter_dataset(process_twitter2017_text, process_twitter2017_image)
    end = time.time()
    print(f"Loading took: {(end - start) * 1000} ms")

async def train_llama_classifier_text():
    model_name = "meta-llama/Llama-3.1-8B"
    print("Loading dataset")
    data, labels = await dataset_text_only()
    print("Dataset loaded")
    token_manager = TokenManager()
    login(token_manager.get_access_token())

    model, tokenizer = create_token_classification_llama(model_name, create_default_quantization_config())
    model = create_parameter_efficient_model(model)

    model = train.training_loop_text_only(model, data["train"], data["val"],tokenizer)
    model.merge_and_unload()
    print("Saving model")
    MODEL_OUT_PATH = "./llama_text.pth"
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    print("Leaving")

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
    #model_name = "meta-llama/Llama-3.1-8B"
    model_name = "mistralai/Mistral-7B-v0.1"
    print("Loading dataset")
    data, labels = await load_twitter_dataset()
    print("Dataset loaded")

    # cnn((image[1]["0_0.jpg"][None,:,:,:])/255)
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    cnn = ConvNet()
    # cnn(torch.rand(3, 256, 256))
    print("Creating model")
    model, tokenizer = create_mistral(model_name, create_default_quantization_config())
    combined = create_parameter_efficient_model(CombinedModel(cnn, model, len(labels.keys())))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], tokenizer)
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
    model, tokenizer = create_llama_model(model_name, create_default_quantization_config())
    combined = CombinedModel(vit, create_parameter_efficient_model(model), len(labels.keys()))
    print("Training combined model")
    combined = train.training_loop_combined(combined, data['train'], data["val"], tokenizer, epochs=5)
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


def draw_dummy_diagram():
    x = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    y = [[1.5788260523967825, 1.313030738320009, 1.2059657401559953, 1.1473848533745683, 1.1107720396701792,
          1.0869205966486934, 1.07328355824383, 1.0652151519363022, 1.0608894993171447, 1.0601054522304882]]
    simple_plot: SimplePlot = PlotBuilder.build_simple_plot(x, y,
                                                            colors=["blue"],
                                                            x_axis_label="epoch",
                                                            y_axis_label="loss",
                                                            plot_title="Vit+Llama training",
                                                            labels=["training_loss"]
                                                            )
    simple_plot.plot()


def conf_matrix_f1():
    y_pred = []
    y_true = []
    for i in range(20):
        l = randint(3, 20)
        y_pred.append(torch.argmax(create_random_tensors((l, 3)), dim=-1))
        y_true.append(create_random_tensors((l,)))
    m = Metrics(y_pred, y_true, 3, {0: "dog", 1: "cat", 2: "horse"})
    matrix = m.confusion_matrix()
    matrix.print_matrix()
    m.f1(matrix, 0)
    print(m.macro_f1(matrix))


if __name__ == "__main__":
    asyncio.run(train_llama_classifier_text())
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
