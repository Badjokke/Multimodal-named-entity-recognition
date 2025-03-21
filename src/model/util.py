import torch
import re
import os
from model.configuration.quantization import peft_from_pretrained


def save_model(model:torch.nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)


def load_lora_model(model, path:str):
    model.load_state_dict(load_and_filter_state_dict(os.path.join(path, "model.pt")))
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        model = peft_from_pretrained(model, path)
    return model

def save_lora_model(model:torch.nn.Module, path:str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    model.save_pretrained(path)

def load_and_filter_state_dict(model_path:str)->dict:
    state_dict = torch.load(model_path, weights_only=True)
    quantization_keywords = ["quant_map", "nested_absmax", "quant_state", "absmax"]
    filtered_state_dict = {}
    for key, value in state_dict.items():
        ignore = False
        for q_keywords in quantization_keywords:
            if re.match(f".*{q_keywords}.*", key) is not None:
                ignore = True
                break
        if not ignore:
            filtered_state_dict[key] = value
    return filtered_state_dict


def create_message(labels: dict, user_text: str)->list[dict[str,str]]:
    return [
        {
            "role" : "system",
            "content": f"You are a bot tasked with named entity recognition. Use these labels: {labels.keys()} a no else."
        },
        {
            "role": "user",
            "content": user_text
        }
    ]


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
