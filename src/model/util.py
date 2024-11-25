import torch
import re


def save_model(model:torch.nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)


def load_and_filter_state_dict(model_path:str)->dict:
    state_dict = torch.load("./combined_model_e1.pth", weights_only=True)
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