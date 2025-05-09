import asyncio

from huggingface_hub import login
from torch import save

from data.text_data_processor.StemmingJsonDataProcessor import StemmingJsonDataProcessor
from data.twitter_loaders.twitter2017_dataset_loader import JsonlDatasetLoader
from model.model_factory import ModelFactory
from model.util import plot_model_training
from security.token_manager import TokenManager
from train import train
from util.ConfigParser import Experiment, ConfigParser
from util.directories_util import DirectoryUtil

# llama is much more sensitive to overfitting due to parameter count
# multimodal models sometimes perform suspiciously poorly
# lower learning rates than bert and bilstm for critical components
llama_learning_rates = {"text_module": 2e-5, "crf": 2e-5, "visual_module": 2e-5, "fusion_layer": 2e-5, "bilstm": 2e-5}

"""==T15 EXPERIMENTS=="""
async def unimodal_image_pipeline_t15(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("image"):
        return
    print("==Running T15 image pipeline==")
    t15_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T15"])
    data, labels, class_occurrences, vocabulary = await t15_loader.load_dataset()
    if exp.contains_model("vit"):
        print("Training vit classifier")
        vit_classifier = ModelFactory.create_vit_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(vit_classifier, data['train'], data["val"], data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/vit/t15/image/fig/plot.png", "ViT classifier T15")
        save(state_dict, model_save_directory + "/vit/t15/image/state_dict/vit_classifier.pth")
    if exp.contains_model("cnn"):
        print("Training cnn classifier")
        cnn_classifier = ModelFactory.create_cnn_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(cnn_classifier, data['train'], data["val"], data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/alex/t15/image/fig/plot.png", "AlexNet classifier T15")
        save(state_dict, model_save_directory + "/alex/t15/image/state_dict/cnn_classifier.pth")
    print("==T15 Image pipeline over---\n")


async def unimodal_text_pipeline_t15(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("text"):
        return
    print("==Running T15 text pipeline==")
    t15_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T15"])
    data, labels, class_occurrences, vocabulary = await t15_loader.load_dataset()
    if exp.contains_model("bert"):
        print("Training bert text only")
        bert_crf, tokenizer = ModelFactory.create_bert_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(bert_crf, data['train'], data["val"], data["test"],
                                                                   tokenizer, class_occurrences, labels, epochs=15,
                                                                   patience=3, text_only=True)
        plot_model_training(results, f"{model_save_directory}/bert/t15/text/fig/plot.png", "Bert+CRF classifier T15")
        save(state_dict, model_save_directory + "/bert/t15/text/state_dict/bert_crf.pth")
    if exp.contains_model("llama"):
        print("Training llama text only")
        llama, tokenizer = ModelFactory.create_llama_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(llama, data['train'],
                                                                   data["val"], data["test"], tokenizer, class_occurrences,
                                                                   labels, epochs=15, patience=3, text_only=True,
                                                                   learning_rates=llama_learning_rates
                                                                   )
        plot_model_training(results, f"{model_save_directory}/llama/t15/text/fig/plot.png", "Llama+CRF classifier T15")
        save(state_dict, model_save_directory + "/llama/t15/text/state_dict/llama_crf_peft.pth")
    if exp.contains_model("lstm"):
        print("Training lstm text only")
        t17_loader = JsonlDatasetLoader(text_processors=[StemmingJsonDataProcessor()])
        data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
        lstm = ModelFactory.create_lstm_text_only_classifier(len(labels.keys()), vocabulary)
        combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                            data["val"], data["test"],
                                                            class_occurrences, labels, patience=5, text_only=True,
                                                            epochs=50)
        plot_model_training(results, f"{model_save_directory}/lstm/t15/text/fig/plot.png",
                            "LSTM CRF text only T15")
        save(state_dict, model_save_directory + "/lstm/t15/text/state_dict/lstm_crf.pth")
    print("==T15 text pipeline over---\n")


async def multimodal_pipeline_t15(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("multimodal"):
        return
    print("==T15 multimodal pipeline start==")
    t17_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T15"])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    if exp.contains_model("bert"):
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t15")
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t15", cnn=True)
    if exp.contains_model("llama"):
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t15")
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t15", cnn=True)
    if exp.contains_model("lstm"):
        t17_loader = JsonlDatasetLoader(text_processors=[StemmingJsonDataProcessor()],
                                        input_path=exp.get_datasets()["T15"])
        data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "t15")
        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "t15", cnn=True)
    print("==T15 multimodal pipeline over==\n")

"""==SOA EXPERIMENTS=="""
async def unimodal_image_pipeline_soa(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("image"):
        return
    print("==Running SOA image pipeline==")
    soa_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["SOA"], include_parent_dir=True,
                                    custom_split=True)
    data, labels, class_occurrences, vocabulary = await soa_loader.load_dataset()
    if exp.contains_model("vit"):
        print("Training vit classifier")
        vit_classifier = ModelFactory.create_vit_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(vit_classifier, data['train'], data["val"],
                                                                  data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/vit/soa/image/fig/plot.png", "ViT classifier SOA")
        save(state_dict, model_save_directory + "/vit/soa/image/state_dict/vit_classifier.pth")
    if exp.contains_model("cnn"):
        print("Training cnn classifier")
        cnn_classifier = ModelFactory.create_cnn_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(cnn_classifier, data['train'], data["val"],
                                                                  data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/alex/soa/image/fig/plot.png", "AlexNet classifier SOA")
        save(state_dict, model_save_directory + "/alex/soa/image/state_dict/cnn_classifier.pth")
    print("==SOA Image pipeline over---\n")


async def unimodal_text_pipeline_soa(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("text"):
        return
    print("==Running SOA text pipeline==")
    soa_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["SOA"], include_parent_dir=True,
                                    custom_split=True)
    data, labels, class_occurrences, vocabulary = await soa_loader.load_dataset()
    if exp.contains_model("bert"):
        print("Training bert text only")
        bert_crf, tokenizer = ModelFactory.create_bert_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(bert_crf, data['train'], data["val"], data["test"],
                                                                   tokenizer, class_occurrences, labels, epochs=15,
                                                                   patience=3, text_only=True)
        plot_model_training(results, f"{model_save_directory}/bert/soa/text/fig/plot.png", "Bert+CRF classifier SOA")
        save(state_dict, model_save_directory + "/bert/soa/text/state_dict/bert_crf.pth")
    if exp.contains_model("llama"):
        print("Training llama text only")
        llama, tokenizer = ModelFactory.create_llama_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(llama, data['train'],
                                                                   data["val"], data["test"], tokenizer,
                                                                   class_occurrences,
                                                                   labels, epochs=15, patience=3, text_only=True,
                                                                   learning_rates=llama_learning_rates
                                                                   )
        plot_model_training(results, f"{model_save_directory}/llama/soa/text/fig/plot.png", "Llama+CRF classifier SOA")
        save(state_dict, model_save_directory + "/llama/soa/text/state_dict/llama_crf_peft.pth")
    if exp.contains_model("lstm"):
        print("Training lstm text only")
        soa_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["SOA"], include_parent_dir=True,
                                        custom_split=True, text_processors=[StemmingJsonDataProcessor()])
        data, labels, class_occurrences, vocabulary = await soa_loader.load_dataset()
        lstm = ModelFactory.create_lstm_text_only_classifier(len(labels.keys()), vocabulary)
        combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                            data["val"], data["test"],
                                                            class_occurrences, labels, patience=5, text_only=True,
                                                            epochs=50)
        plot_model_training(results, f"{model_save_directory}/lstm/t15/text/fig/plot.png",
                            "LSTM CRF text only T15")
        save(state_dict, model_save_directory + "/lstm/t15/text/state_dict/lstm_crf.pth")
        print("==SOA text pipeline over---\n")


async def multimodal_pipeline_soa(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("multimodal"):
        return
    print("==SOA multimodal pipeline start==")
    t17_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["SOA"], include_parent_dir=True,
                                    custom_split=True)
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    if exp.contains_model("bert"):
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "soa")
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "soa", cnn=True)
    if exp.contains_model("llama"):
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "soa")
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "soa", cnn=True)
    if exp.contains_model("lstm"):
        t17_loader = JsonlDatasetLoader(text_processors=[StemmingJsonDataProcessor()],
                                        input_path=exp.get_datasets()["SOA"], include_parent_dir=True,
                                        custom_split=True)
        data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "soa")
        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "soa", cnn=True)

    print("==SOA multimodal pipeline over==\n")


"""==T17 EXPERIMENTS=="""
async def unimodal_image_pipeline_t17(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("image"):
        return
    print("==Running T17 image pipeline==")
    t17_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T17"])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    if exp.contains_model("vit"):
        print("Training vit classifier")
        vit_classifier = ModelFactory.create_vit_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(vit_classifier, data['train'], data["val"],
                                                                  data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/vit/t17/image/fig/plot.png", "ViT classifier T17")
        save(state_dict, model_save_directory + "/vit/t17/image/state_dict/vit_classifier.pth")
        print()
    if exp.contains_model("cnn"):
        print("Training cnn classifier")
        cnn_classifier = ModelFactory.create_cnn_classifier(len(labels.keys()))
        combined, results, state_dict = train.image_only_training(cnn_classifier, data['train'], data["val"],
                                                                  data["test"],
                                                                  class_occurrences, labels, epochs=15, patience=3)
        plot_model_training(results, f"{model_save_directory}/alex/t17/image/fig/plot.png", "AlexNet classifier T17")
        save(state_dict, model_save_directory + "/alex/t17/image/state_dict/cnn_classifier.pth")
    print("==T17 Image pipeline over---\n")


async def unimodal_text_pipeline_t17(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("text"):
        return
    print("==Running T17 text pipeline==")
    t17_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T17"])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    if exp.contains_model("bert"):
        print("Training bert text only")
        bert_crf, tokenizer = ModelFactory.create_bert_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(bert_crf, data['train'], data["val"], data["test"],
                                                                   tokenizer, class_occurrences, labels, epochs=15,
                                                                   patience=3, text_only=True)
        plot_model_training(results, f"{model_save_directory}/bert/t17/text/fig/plot.png", "Bert+CRF classifier T17")
        save(state_dict, model_save_directory + "/bert/t17/text/state_dict/bert_crf.pth")
    if exp.contains_model("llama"):
        print("Training llama text only")
        llama, tokenizer = ModelFactory.create_llama_text_only_classifier(len(labels.keys()))
        combined, results, state_dict = train.transformer_training(llama, data['train'],
                                                                   data["val"], data["test"], tokenizer,
                                                                   class_occurrences,
                                                                   labels, epochs=15, patience=3, text_only=True,
                                                                   learning_rates=llama_learning_rates)
        plot_model_training(results, f"{model_save_directory}/llama/t17/text/fig/plot.png", "Llama+CRF classifier T17")
        save(state_dict, model_save_directory + "/llama/t17/text/state_dict/llama_crf_peft.pth")
    if exp.contains_model("lstm"):
        print("Training lstm text only")
        t17_loader = JsonlDatasetLoader(text_processors=[StemmingJsonDataProcessor()])
        data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
        lstm = ModelFactory.create_lstm_text_only_classifier(len(labels.keys()), vocabulary)
        combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                            data["val"], data["test"],
                                                            class_occurrences, labels, patience=5, text_only=True,
                                                            epochs=50)
        plot_model_training(results, f"{model_save_directory}/lstm/t17/text/fig/plot.png",
                            "LSTM CRF text only T17")
        save(state_dict, model_save_directory + "/lstm/t17/text/state_dict/lstm_crf.pth")
    print("==T17 text pipeline over---\n")


async def multimodal_pipeline_t17(model_save_directory: str, exp: Experiment):
    if not exp.contains_pipeline("multimodal"):
        return
    print("==T17 multimodal pipeline start==")
    t17_loader = JsonlDatasetLoader(input_path=exp.get_datasets()["T17"])
    data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()
    if exp.contains_model("bert"):
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t17")
        bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t17", cnn=True)
    if exp.contains_model("llama"):
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t17")
        llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, "t17", cnn=True)
    if exp.contains_model("lstm"):
        t17_loader = JsonlDatasetLoader(text_processors=[StemmingJsonDataProcessor()])
        data, labels, class_occurrences, vocabulary = await t17_loader.load_dataset()

        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "t17")
        lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, "t17", cnn=True)

    print("==T17 multimodal pipeline over==\n")


def bert_multimodal_t17(model_save_directory, data, labels, class_occurrences, dataset, cnn=False):
    label_count = len(labels.keys())
    visual_model_label = "VIT" if not cnn else "CNN"

    print(f"==BERT_{visual_model_label}_CROSS_ATTENTION==")
    bert_vit, tokenizer = ModelFactory.create_bert_vit_attention_classifier(
        label_count) if not cnn else ModelFactory.create_bert_cnn_attention_classifier(label_count)
    combined, results, state_dict = train.transformer_training(bert_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3)
    plot_model_training(results, f"{model_save_directory}/bert/{dataset}/multimodal/fig/cross_plot.png",
                        f"Cross Attention Bert+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/bert/{dataset}/multimodal/state_dict/bert_{visual_model_label.lower()}_cross_attention.pth")

    print(f"==BERT_{visual_model_label}_LINEAR_FUSION==")
    bert_vit, tokenizer = ModelFactory.create_bert_vit_linear_fusion(
        label_count) if not cnn else ModelFactory.create_bert_cnn_linear_fusion(label_count)
    combined, results, state_dict = train.transformer_training(bert_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3)
    plot_model_training(results, f"{model_save_directory}/bert/{dataset}/multimodal/fig/linear_plot.png",
                        f"Linear fusion Bert+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/bert/{dataset}/multimodal/state_dict/bert_{visual_model_label.lower()}_linear_fusion.pth")
    print(f"==BERT_{visual_model_label}_PARTIAL_PREDICTION==")
    bert_vit, tokenizer = ModelFactory.create_bert_vit_partial_prediction(
        label_count) if not cnn else ModelFactory.create_bert_cnn_partial_prediction(label_count)
    combined, results, state_dict = train.transformer_training(bert_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3,
                                                               cross_loss=True)
    plot_model_training(results, f"{model_save_directory}/bert/{dataset}/multimodal/fig/partial_plot.png",
                        f"Partial Prediction BERT+{visual_model_label} classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/bert/{dataset}/multimodal/state_dict/bert_{visual_model_label.lower()}_partial_prediction.pth")


def llama_multimodal_t17(model_save_directory, data, labels, class_occurrences, dataset, cnn=False):
    label_count = len(labels.keys())
    visual_model_label = "VIT" if not cnn else "CNN"
    print(f"==LLAMA_{visual_model_label}_CROSS_ATTENTION==")
    llama_vit, tokenizer = ModelFactory.create_llama_vit_attention_classifier(
        label_count) if not cnn else ModelFactory.create_llama_cnn_attention_classifier(label_count)
    combined, results, state_dict = train.transformer_training(llama_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3,
                                                               learning_rates=llama_learning_rates)
    plot_model_training(results, f"{model_save_directory}/llama/{dataset}/multimodal/fig/cross_plot.png",
                        f"Cross Attention Llama+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/llama/{dataset}/multimodal/state_dict/llama_{visual_model_label.lower()}_cross_attention_peft.pth")
    print(f"==LLAMA_{visual_model_label}_LINEAR_FUSION==")
    llama_vit, tokenizer = ModelFactory.create_llama_vit_linear_fusion(
        label_count) if not cnn else ModelFactory.create_llama_cnn_linear_fusion(label_count)
    combined, results, state_dict = train.transformer_training(llama_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3,
                                                               learning_rates=llama_learning_rates)
    plot_model_training(results, f"{model_save_directory}/llama/{dataset}/multimodal/fig/linear_plot.png",
                        f"Linear fusion Llama+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/llama/{dataset}/multimodal/state_dict/llama_{visual_model_label.lower()}_linear_fusion_peft.pth")
    print(f"==LLAMA_{visual_model_label}_PARTIAL_PREDICTION==")
    llama_vit, tokenizer = ModelFactory.create_llama_vit_partial_prediction(
        label_count) if not cnn else ModelFactory.create_llama_cnn_partial_prediction(label_count)
    combined, results, state_dict = train.transformer_training(llama_vit, data['train'], data["val"], data["test"],
                                                               tokenizer,
                                                               class_occurrences, labels, epochs=15, patience=3,
                                                               cross_loss=True, learning_rates=llama_learning_rates)
    plot_model_training(results, f"{model_save_directory}/llama/{dataset}/multimodal/fig/partial_plot.png",
                        f"Llama+{visual_model_label} Partial prediction classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/llama/{dataset}/multimodal/state_dict/llama_{visual_model_label.lower()}_partial_prediction_peft.pth")


def lstm_multimodal_t17(model_save_directory, data, labels, class_occurrences, vocabulary, dataset, cnn=False):
    label_count = len(labels.keys())
    visual_model_label = "VIT" if not cnn else "CNN"
    print(f"==LSTM_{visual_model_label}_CROSS_ATTENTION==")
    lstm = ModelFactory.create_lstm_vit_attention_classifier(label_count,
                                                             vocabulary) if not cnn else ModelFactory.create_lstm_cnn_attention_classifier(
        label_count, vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                        data["val"], data["test"],
                                                        class_occurrences, labels, patience=5)
    plot_model_training(results, f"{model_save_directory}/lstm/{dataset}/multimodal/fig/cross_plot.png",
                        f"Cross Attention BILSTM+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/lstm/{dataset}/multimodal/state_dict/lstm_{visual_model_label.lower()}_cross_attention.pth")
    print(f"==LSTM_{visual_model_label}_LINEAR_FUSION==")
    lstm = ModelFactory.create_lstm_vit_linear_fusion(label_count,
                                                      vocabulary) if not cnn else ModelFactory.create_lstm_cnn_linear_fusion(
        label_count, vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                        data["val"], data["test"],
                                                        class_occurrences, labels, patience=5)
    plot_model_training(results, f"{model_save_directory}/lstm/{dataset}/multimodal/fig/linear_plot.png",
                        f"Linear fusion BILSTM+{visual_model_label}+CRF classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/lstm/{dataset}/multimodal/state_dict/lstm_{visual_model_label.lower()}_linear.pth")
    print(f"==LSTM_{visual_model_label}_PARTIAL_PREDICTION==")
    lstm = ModelFactory.create_lstm_vit_partial_prediction(label_count,
                                                           vocabulary) if not cnn else ModelFactory.create_lstm_cnn_partial_prediction(
        label_count, vocabulary)
    combined, results, state_dict = train.lstm_training(lstm, data['train'],
                                                        data["val"], data["test"],
                                                        class_occurrences, labels, patience=5, cross_loss=True)
    plot_model_training(results, f"{model_save_directory}/lstm/{dataset}/multimodal/fig/partial_plot.png",
                        f"BILSTM+{visual_model_label} Partial prediction classifier {dataset.upper()}")
    save(state_dict,
         f"{model_save_directory}/lstm/{dataset}/multimodal/state_dict/lstm_{visual_model_label.lower()}_partial.pth")


def login_user():
    token_manager = TokenManager()
    login(token_manager.get_access_token())


def load_experiments(config_file_path: str) -> list[Experiment]:
    return ConfigParser(config_file_path).get_experiments()

def create_experiment_directories(exp: Experiment):
    base_path = exp.results_path
    util = DirectoryUtil(base_path, set(exp.get_datasets().keys()), set(exp.get_models()), set(exp.get_pipeline()))
    util.create_directories_for_result_logs()

if __name__ == "__main__":
    experiments = load_experiments("experiments_config.json")
    login_user()
    for idx, experiment in enumerate(experiments):
        print(f"==Experiment {idx} START==")
        base_path = experiment.get_results_path()
        create_experiment_directories(experiment)
        if experiment.contains_dataset("soa"):
            asyncio.run(unimodal_image_pipeline_soa(base_path, experiment))
            asyncio.run(unimodal_text_pipeline_soa(base_path, experiment))
            asyncio.run(multimodal_pipeline_soa(base_path, experiment))
        if experiment.contains_dataset("t15"):
            asyncio.run(unimodal_image_pipeline_t15(base_path, experiment))
            asyncio.run(unimodal_text_pipeline_t15(base_path, experiment))
            asyncio.run(multimodal_pipeline_t15(base_path, experiment))
        if experiment.contains_dataset("t17"):
            asyncio.run(unimodal_image_pipeline_t17(base_path, experiment))
            asyncio.run(unimodal_text_pipeline_t17(base_path, experiment))
            asyncio.run(multimodal_pipeline_t17(base_path, experiment))
        print(f"==Experiment {idx} END==")
