from transformers import AutoModelForCausalLM, AutoTokenizer
import security.token_manager as token_manager
from peft import PeftModel, prepare_model_for_kbit_training
from torch import float16
from model.quantization_config import(create_lora_config,find_all_linear_names)

def create_base_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        token=token_manager.get_access_token()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_manager.get_access_token())
    
    tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer


def create_quant_model(model_name,bnb, new_model):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=float16,
        quantization_config=bnb,
        device_map="auto",
        token=token_manager.get_access_token()
    )
    model = prepare_model_for_kbit_training(model=model,use_gradient_checkpointing=True)
    #model = PeftModel.from_pretrained(model, token=token_manager.get_access_token(),config=create_lora_config(find_all_linear_names(model)))
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_manager.get_access_token())
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

