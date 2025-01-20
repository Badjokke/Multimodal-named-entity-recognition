from typing import Iterable, Union

import numpy as np
import torch
from peft import PeftModel
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _compute_inverse_freq_weights(y, alpha=0.5) -> torch.Tensor:
    weights = [1 / (np.bincount(x) ** alpha) for x in y]
    return torch.Tensor(weights).to(torch.float32).to(device)

def _compute_class_weights_rare_events(y) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    return torch.from_numpy(weights).to(torch.float32).to(device)

def _create_cross_entropy_loss_criterion(y) -> torch.nn.CrossEntropyLoss:
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    w = torch.from_numpy(weights).to(torch.float32).to(device)
    return torch.nn.CrossEntropyLoss(ignore_index=-100, weight=w)

def align_labels(word_ids, labels):
    padded_labels = []
    for word in word_ids:
        padded_labels.append(-100 if word is None else labels[word])
    return padded_labels

def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)

def _create_adamw_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate,weight_decay=0.01)

def _create_scheduler(optimizer, t_max) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, verbose=True)



def setup_optimizer(model, t_max, lr=1e-5, weight_decay=0.1, warmup_steps=1000):
    # Initialize AdamW optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=lr,  # Learning rate
        betas=(0.9, 0.999),  # Beta parameters
        eps=1e-8,  # Epsilon parameter
        weight_decay=weight_decay
    )

    # Add linear warmup scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_max
    )

    return optimizer, scheduler
def training_loop_text_only(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer, epochs=10,
                            patience=3):
    model.to(device)
    model.train()

    loss_criterion = _create_cross_entropy_loss_criterion(None)
    optimizer = _create_optimizer(model.parameters())

    mean_loss = 0

    for epoch in range(epochs):
        loss = perform_epoch_text_only(model, tokenizer, train_data, loss_criterion, optimizer)
        loss = loss / len(train_data)
        print(f"[epoch: {epoch + 1}] Training loss: {loss}")
        mean_loss += loss
    print(f"Average loss: {mean_loss / epochs}")
    return model

def perform_epoch_text_only(model, tokenizer, train_data, loss_criterion, optimizer):
    running_loss = 0.0
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text, labels = tokenizer(data_sample[0], return_tensors="pt", padding=True, truncation=True,
                                 is_split_into_words=True), data_sample[1].to(device)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(text)
        loss = loss_criterion(outputs.squeeze(0), aligned_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        running_loss += loss.item()

        optimizer.step()

    return running_loss


def training_loop_combined_lstm(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, vocabulary,
                                class_occurrences, epochs=10, patience=3):
    model.to(device)
    loss_criterion = _create_cross_entropy_loss_criterion(class_occurrences)
    optimizer = _create_adamw_optimizer(model.parameters())
    scheduler = _create_scheduler(optimizer, t_max=epochs * len(train_data))

    for epoch in range(epochs):
        training_loss = perform_epoch_combined_lstm(model, vocabulary, train_data, loss_criterion, optimizer, scheduler)
        val_loss = validate_after_epoch(model, vocabulary, loss_criterion, validation_data)
        print(f"[epoch: {epoch + 1}] Training loss: {training_loss}")
        print(f"[epoch: {epoch + 1}] Validation loss: {val_loss}")

def perform_epoch_combined_lstm(model, vocabulary, train_data, loss_criterion, optimizer, scheduler):
    model.train()
    for i in range(len(train_data)):
        text, images, labels = torch.tensor(list(map(lambda word: vocabulary[word],train_data[i][0])), dtype=torch.long),train_data[i][1].to(device), torch.tensor(train_data[i][2], dtype=torch.long, device=device)
        outputs = model(images, text)
        loss = loss_criterion(outputs.squeeze(0), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
        optimizer.step()
        scheduler.step()

def perform_validation_combined_lstm(model, vocabulary, loss_criterion, validation_data):
    model.eval()
    loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(validation_data)):
            data_sample = validation_data[i]
            text, images, labels = torch.Tensor(list(map(lambda word: vocabulary[word], data_sample[0])),device=device),torch.tensor(data_sample[2],dtype=torch.long, device=device) ,data_sample[1].to(device)
            outputs = model(images, text)
            loss += loss_criterion(outputs.squeeze(0), labels)
            y_true.append(labels)
            y_pred.append(torch.argmax(outputs.squeeze(0), dim=1))
    return loss / len(validation_data)

def training_loop_combined(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer,
                           class_occurrences, epochs=10, patience=3):
    model.to(device)
    loss_criterion = _create_cross_entropy_loss_criterion(class_occurrences)
    #optimizer = _create_adamw_optimizer(model.parameters())
    #scheduler = _create_scheduler(optimizer, t_max=epochs * len(train_data))
    optimizer, scheduler = setup_optimizer(model, t_max=epochs*len(train_data))
    for epoch in range(epochs):
        training_loss = perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler)
        val_loss = validate_after_epoch(model, tokenizer, loss_criterion, validation_data)
        print(f"[epoch: {epoch + 1}] Training loss: {training_loss}")
        print(f"[epoch: {epoch + 1}] Validation loss: {val_loss}")
    return model

def early_stop():
    print("Implement me!")

def perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    for i in range(len(train_data)):
        data_sample = train_data[i]
        images, labels, text = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,
                                                                       device=device), tokenizer(data_sample[0],
                                                                                                 return_tensors="pt",
                                                                                                 is_split_into_words=True)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long)

        outputs = model(images, text)
        loss = loss_criterion(outputs.squeeze(0), aligned_labels)
        running_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


    return running_loss / len(train_data)

def validate_after_epoch(model, tokenizer, loss_criterion, validation_data) -> tuple[
    float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    model.eval()
    loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(validation_data)):
            data_sample = validation_data[i]
            images, labels, text = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,
                                                                           device=device), tokenizer(data_sample[0],
                                                                                                     return_tensors="pt",
                                                                                                     is_split_into_words=True)
            labels = torch.tensor(align_labels(text.word_ids(), labels), device=device)
            text = {key: value.to(device) for key, value in text.items()}

            outputs = model(images, text).squeeze(0)
            loss += loss_criterion(outputs, labels)
            y_true.append(labels)
            y_pred.append(torch.argmax(outputs, dim=1))
    return loss / len(validation_data) #(y_true, y_pred)

def get_occurrence_count(l: list):
    dic = dict()
    for n in l:
        if n not in dic:
            dic[n] = 0
        dic[n] += 1
    return dic

def get_index_of_max_value(dic: dict[int, int]):
    v = 0
    i = 0
    for (key, value) in dic.items():
        if value > v:
            v = value
            i = key
    return i

def decode_labels_majority_vote(word_ids, y_predicted):
    decoded_labels = []
    walker = 0
    while walker < len(word_ids) - 1:
        current_w_id = word_ids[walker]
        if current_w_id == -100:
            continue
        start_index = walker
        while word_ids[walker + 1] == current_w_id:
            walker += 1
        walker += 1
        decoded_labels.append(get_index_of_max_value(get_occurrence_count(y_predicted[start_index:walker])))
    return torch.tensor(decoded_labels, dtype=torch.long, device=device)

def inference_loop_combined_model(model: torch.nn.Module, test_data, tokenizer):
    model.to(device)
    model.eval()
    y_predicted = []
    y_actual = []
    for i in range(len(test_data)):
        sample = test_data[i]
        images = sample[1].to(device)
        labels = sample[2].to(device)
        text = tokenizer(sample[0], return_tensors="pt", is_split_into_words=True)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        outputs = model(images, text)
        y_pred = decode_labels_majority_vote(word_ids, torch.argmax(outputs.squeeze(0), dim=1))
        y_predicted.append(y_pred)
        y_actual.append(labels)
    return y_actual, y_predicted
