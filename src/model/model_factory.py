from torch import float16
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM, \
    RobertaTokenizerFast, RobertaModel, ViTImageProcessor, ViTModel

from model.visual.convolutional_net import ConvNet


# label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
# id2label = {v: k for k, v in label2id.items()}

def create_vit():
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    return model, processor

def create_convolutional_net():
    return ConvNet()

def create_roberta_base():
    model = RobertaModel.from_pretrained("roberta-base", device_map="auto", low_cpu_mem_usage=True, torch_dtype=float16)
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    return model, tokenizer

def create_model(model_name, bnb):
    model = LlamaModel.from_pretrained(
        model_name,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=float16,
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
        torch_dtype=float16,
        quantization_config=bnb,
        # num_labels=len(label2id.items()),
        # id2label=id2label,
        # label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
