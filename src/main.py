from huggingface_hub import login

from data.dataset import (load_hf_dataset)
from data.dataset_util import preprocess_dataset_class
from model.factory.model_factory import (create_model)
from model.quantization_config import create_default_quantization_config
from model.train.trainer import (train_model)
from security.token_manager import get_access_token


def ner_classify():
    model_name = "mistralai/Mistral-7B-v0.1"
    model, tokenizer = create_model(model_name, None)
    ds = preprocess_dataset_class(load_hf_dataset(), tokenizer)

    train_model(model, tokenizer, ds)


if __name__ == "__main__":
    login(get_access_token())
    ner_classify()
