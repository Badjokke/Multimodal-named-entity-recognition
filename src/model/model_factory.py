from torch import float16
from transformers import LlamaForTokenClassification, LlamaTokenizerFast, AutoModelForTokenClassification, \
    AutoTokenizer, BitsAndBytesConfig, LlamaModel
from model.custom_models.convolutional_net import ConvNet
from typing import Callable
#label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
#id2label = {v: k for k, v in label2id.items()}


def create_llama_classifier(model_name, bnb):
    model = LlamaForTokenClassification.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=float16,
        quantization_config=bnb,
        #num_labels=len(label2id.items()),
        #id2label=id2label,
        #label2id=label2id
    )
    llamaTokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    llamaTokenizer.pad_token = llamaTokenizer.eos_token
    return model, llamaTokenizer


def create_convolutional_net():
    return ConvNet()


def create_model(model_name, bnb):
    model = LlamaModel.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=float16,
        quantization_config=bnb,
        #num_labels=len(label2id.items()),
        #id2label=id2label,
        #label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def create_multimodal_model(text_model_supplier:Callable[[str, BitsAndBytesConfig], str], visual_model_supplier: Callable[[str], str]):
    pass