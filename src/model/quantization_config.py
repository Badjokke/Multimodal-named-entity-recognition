from transformers import BitsAndBytesConfig
from torch import bfloat16
from peft import LoraConfig
import bitsandbytes as bnb


def create_default_quantization_config():
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
    )
    return quantization_config


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def create_lora_config(modules):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
                r=8,  # dimension of the updated matrices
                lora_alpha=32,  # parameter for scaling
                target_modules=modules,
                lora_dropout=0.05,  # dropout probability for layers
                bias="none",
                task_type="TOKEN_CLS",
    )
    return config
