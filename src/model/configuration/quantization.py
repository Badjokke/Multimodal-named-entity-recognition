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
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    """
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj'], # this allows us to adjust weights for matrices used in attention computation
    )
    return config


def create_parameter_efficient_model(model) -> PeftModel:
    return get_peft_model(model, _create_lora_config())