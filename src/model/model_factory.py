from torch import float16, bfloat16,float32
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM, \
    RobertaTokenizerFast, RobertaModel, ViTImageProcessor, ViTModel, MistralModel, LlamaTokenizerFast, BertTokenizerFast, BertModel
from model.visual.convolutional_net import ConvNet
from model.language.lstm import LSTM
from model.configuration.quantization import create_parameter_efficient_model
from model.language.transformer_classifier import TransformerClassifier

def create_vit():
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    return model, processor

def create_convolutional_net():
    return ConvNet()

def create_lstm(vocab_size, bidirectional=True):
    return LSTM(vocab_size,bidirectional)

def create_bert_large():
    tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased', device_map="auto", low_cpu_mem_usage=True, torch_dtype=float32)
    model = BertModel.from_pretrained("bert-large-uncased")
    return model, tokenizer

def create_llama_model(model_name, bnb):
    model = LlamaModel.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=float32,
        quantization_config=bnb
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def create_mistral(model_name, bnb):
    model = MistralModel.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=bfloat16,
        quantization_config=bnb,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def create_model_for_lm(model_name, bnb):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=bfloat16,
        quantization_config=bnb,
    )
    tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # Additional options specific to fast tokenizer
            add_prefix_space=True,  # Important for proper tokenization of first word
            use_fast=True,         # Explicitly enable fast tokenization
        )
    return model, tokenizer
