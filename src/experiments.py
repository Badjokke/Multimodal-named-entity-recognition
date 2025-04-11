import asyncio
import time
from typing import Callable, Coroutine

from torch import save
from huggingface_hub import login

from data.dataset_analyzer import DatasetAnalyzer
from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from data.twitter_preprocessors.twitter2015_preprocessor import Twitter2015Preprocessor
from data.twitter_preprocessors.twitter2017_preprocessor import Twitter2017Preprocessor
from data.text_data_processor.stemming_json_data_processor import StemmingTextDataProcessor
from metrics.plot_builder import PlotBuilder
from model.model_factory import ModelFactory
from model.util import plot_model_training
from security.token_manager import TokenManager
from model.configuration.quantization import create_parameter_efficient_model
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


async def unimodal_image_pipeline_t17(model_save_directory:str):
    print("Training lstm+vit")
    t17_loader = JsonlDatasetLoader(text_processors=[StemmingTextDataProcessor()])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    lstm = ModelFactory.create_lstm_vit_attention_classifier(len(labels.keys()), vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                               data["val"], data["test"],
                                                               class_occurrences, labels, patience=5)
    plot_model_training(results, f"{model_save_directory}/lstm/t17/fig/plot.png",
                        "Multimodal Cross Attention LSTM with CRF")
    save(state_dict, model_save_directory + "/lstm_vit_cross_attention_peft.pth")





    print("Running T17 unimodal pipeline")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    print("Training vit classifier")
    vit_classifier = ModelFactory.create_vit_classifier(len(labels.keys()))
    combined, results, state_dict = train.image_only_training(vit_classifier, data['train'], data["val"], data["test"], class_occurrences, labels, epochs=15,patience=3)
    plot_model_training(results, f"{model_save_directory}/vit/t17/fig/plot.png","ViT classifier")
    save(state_dict, model_save_directory + "/vit_classifier.pth")
    print()
    print("Training cnn classifier")
    cnn_classifier = ModelFactory.create_cnn_classifier(len(labels.keys()))
    combined, results, state_dict = train.image_only_training(cnn_classifier, data['train'], data["val"], data["test"], class_occurrences, labels, epochs=15,patience=3)
    plot_model_training(results, f"{model_save_directory}/alex/t17/fig/plot.png","AlexNet classifier")
    save(state_dict, model_save_directory + "/cnn_classifier.pth")
    print("---image pipeline over---\n")


async def unimodal_text_pipeline_t17(model_save_directory:str):
    print("Running T17 unimodal pipeline")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    print("Training bert text only")
    bert_crf, tokenizer = ModelFactory.create_bert_text_only_classifier(len(labels.keys()))
    combined, results, state_dict = train.transformer_training(bert_crf, data['train'], data["val"], data["test"],tokenizer, class_occurrences, labels, epochs=15,patience=3, text_only=True)
    plot_model_training(results, f"{model_save_directory}/bert/t17/text/fig/plot.png","BERT crf text only")
    save(state_dict, model_save_directory + "/bert_crf.pth")
    print()
    print("Training llama text only")
    llama, tokenizer = ModelFactory.create_llama_text_only_classifier(len(labels.keys()))
    combined, results, state_dict = train.transformer_training(create_parameter_efficient_model(llama), data['train'], data["val"], data["test"],tokenizer, class_occurrences, labels, epochs=15,patience=3, text_only=True)
    plot_model_training(results, f"{model_save_directory}/llama/t17/text/fig/plot.png","Llama crf text only")
    save(state_dict, model_save_directory + "/bert_crf.pth")
    print()
    print("Training lstm+vit")
    t17_loader = JsonlDatasetLoader(text_processors=[StemmingTextDataProcessor()])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    lstm = ModelFactory.create_lstm_vit_attention_classifier(len(labels.keys()), vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                        data["val"], data["test"],
                                                        class_occurrences, labels, patience=5)
    plot_model_training(results, f"{model_save_directory}/lstm/t17/fig/plot.png",
                        "Multimodal Cross Attention LSTM with CRF")
    save(state_dict, model_save_directory + "/lstm_vit_cross_attention_peft.pth")
    print("---text pipeline over---\n")

async def multimodal_pipeline_t17(model_save_directory: str):
    print("Running T17 multimodal pipeline")
    t17_loader = JsonlDatasetLoader()
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    #bert_vit, tokenizer = ModelFactory.create_bert_vit_attention_classifier(len(labels.keys()))
    bert_vit, tokenizer = ModelFactory.create_bert_vit_partial_prediction(len(labels.keys()))
    print("Training bert+vit")
    combined, results, state_dict = train.transformer_training(bert_vit, data['train'], data["val"], data["test"], tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=2, cross_loss=True)
    plot_model_training(results, f"{model_save_directory}/bert/t17/fig/plot.png", "Multimodal Cross Attention BERT with CRF")
    save(state_dict, f"{model_save_directory}/bert/t17/bert_vit_cross_attention.pth")
    print()
    print("Training llama+vit")
    llama, llama_tokenizer = ModelFactory.create_llama_vit_attention_classifier(len(labels.keys()))
    combined, results, state_dict = train.transformer_training(create_parameter_efficient_model(llama), data['train'], data["val"], data["test"], llama_tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=2)
    plot_model_training(results, f"{model_save_directory}/llama/t17/fig/plot.png", "Multimodal Cross Attention Llama with CRF")
    save(state_dict, f"{model_save_directory}/bert/t17/llama_vit_cross_attention_peft.pth")

    print()

    print("Training lstm+vit")
    t17_loader = JsonlDatasetLoader(text_processors=[StemmingTextDataProcessor()])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    lstm = ModelFactory.create_lstm_vit_attention_classifier(len(labels.keys()),vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                               data["val"], data["test"],
                                                               class_occurrences, labels, patience=5)
    plot_model_training(results, f"{model_save_directory}/lstm/t17/fig/plot.png",
                        "Multimodal Cross Attention LSTM with CRF")
    save(state_dict, f"{model_save_directory}/bert/t17//lstm_vit_cross_attention_peft.pth")

    print("---multimodal pipeline over---\n")

async def multimodal_pipeline_t15(model_save_directory: str):
    print("Running T15 multimodal pipeline")
    t17_loader = JsonlDatasetLoader(input_path="../dataset/preprocessed/twitter_2015")
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()

    bert_vit, tokenizer = ModelFactory.create_bert_vit_attention_classifier(len(labels.keys()))
    print("Training bert+vit")
    combined, results, state_dict = train.transformer_training(bert_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=10, patience=2)
    plot_model_training(results, f"{model_save_directory}/bert/t15/fig/plot.png", "Multimodal Cross Attention BERT with CRF")
    save(state_dict, model_save_directory + "/bert_vit_cross_attention.pth")
    print()
    print("Training llama+vit")
    llama, llama_tokenizer = ModelFactory.create_llama_vit_attention_classifier(len(labels.keys()))
    combined, results, state_dict = train.transformer_training(create_parameter_efficient_model(llama), data['train'],
                                                               data["val"], data["test"], llama_tokenizer,
                                                               class_occurrences, labels, epochs=10, patience=2)
    plot_model_training(results, f"{model_save_directory}/llama/t15/fig/plot.png", "Multimodal Cross Attention Llama with CRF")
    save(state_dict, model_save_directory + "/llama_vit_cross_attention_peft.pth")
    print()
    print("Leaving")

if __name__ == "__main__":
    token_manager = TokenManager()
    login(token_manager.get_access_token())
    # asyncio.run(preprocess_twitter15())
    #asyncio.run(llama_vit_multimodal())
    # asyncio.run(preprocess_twitter15())
    # t17_loader = JsonlDatasetLoader(lightweight=True)
    print("== running T17 STUFF ==")
    #asyncio.run(unimodal_image_pipeline_t17("../models"))
    #asyncio.run(unimodal_text_pipeline_t17("../models"))
    asyncio.run(multimodal_pipeline_t17("../models"))
    print("== running T15 STUFF ==")
    asyncio.run(multimodal_pipeline_t15("../models"))
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
