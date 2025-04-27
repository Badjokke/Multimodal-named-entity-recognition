from peft import LoraConfig, TaskType, get_peft_model, PeftModel
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


def _create_lora_config() -> LoraConfig:
    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="TOKEN_CLS"
    )
    return config


def create_parameter_efficient_model(model) -> PeftModel:
    return get_peft_model(model, _create_lora_config())

def peft_from_pretrained(model, path) -> PeftModel:
    return PeftModel.from_pretrained(model, path)