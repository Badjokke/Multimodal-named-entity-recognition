from peft import LoraConfig, TaskType
from torch import bfloat16
from transformers import BitsAndBytesConfig


def create_default_quantization_config() -> BitsAndBytesConfig:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    return quantization_config


def create_lora_config() -> LoraConfig:
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=2,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type=TaskType.TOKEN_CLS,
    )
    return config
