import os
import re

import torch

from metrics.plot_builder import PlotBuilder
from model.configuration.quantization import peft_from_pretrained


def save_model(model: torch.nn.Module, model_path: str):
    torch.save(model.state_dict(), model_path)


def load_lora_model(model, path: str):
    model.load_state_dict(load_and_filter_state_dict(os.path.join(path, "model.pt")))
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        model = peft_from_pretrained(model, path)
    return model


def save_lora_model(model: torch.nn.Module, path: str):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    model.save_pretrained(path)


def load_and_filter_state_dict(model_path: str) -> dict:
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


def plot_model_training(results: list[dict[str,tuple]], path: str, plot_title: str):
    print("Plotting training results")
    x = []
    y_train = []
    y_val = []
    y_test = []
    for i in range(len(results)):
        x.append(i)
        y_train.append((results[i]["train"][1]["macro"]) * 100)
        y_val.append((results[i]["val"][1]["macro"]) * 100)
        y_test.append((results[i]["test"][1]["macro"]) * 100)
    plot = PlotBuilder.build_simple_plot([x] * 3, [y_train, y_val, y_test], **{"plot_title":plot_title, "x_axis_label": "epochs", "y_axis_label": "macro f1", "labels":["train f1 macro", "validation f1 macro", "test f1 macro"]})
    plot.plot()
    print(f"Saving plot to {path}")
    plot.save(path)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model
