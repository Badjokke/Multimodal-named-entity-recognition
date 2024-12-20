from typing import Iterable, Union

import torch
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_cross_entropy_loss_criterion(labels) -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss(ignore_index=-100)


def align_labels(word_ids, labels):
    padded_labels = []
    for word in word_ids:
        padded_labels.append(-100 if word is None else labels[word])
    return padded_labels

def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=0.01)

def _create_adamw_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate)

def _create_scheduler(optimizer, t_max) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def training_loop_text_only(model:Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer, epochs=10, patience=3 ):
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
    print(f"Average loss: {mean_loss/epochs}")
    return model


def perform_epoch_text_only(model, tokenizer, train_data, loss_criterion, optimizer):
    running_loss = 0.0
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text, labels =  tokenizer(data_sample[0],return_tensors="pt",padding=True, truncation=True, is_split_into_words=True), data_sample[1].to(device)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(text)
        loss = loss_criterion(outputs.squeeze(0), aligned_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        running_loss += loss.item()

        optimizer.step()

    return running_loss

def training_loop_combined(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer, epochs=10, patience=3):
    model.to(device)
    model.train()

    loss_criterion = _create_cross_entropy_loss_criterion(None)
    optimizer = _create_optimizer(model.parameters())
    scheduler = _create_scheduler(optimizer, t_max=epochs * len(train_data))

    mean_loss = 0
    previous_loss = float("inf")
    no_improvement_counter = 0
    for epoch in range(epochs):
        if no_improvement_counter > patience:
            print(f"Early stopping after {epoch+1} epochs. Loss: {mean_loss}")
            early_stop()

        loss = perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler)
        loss = loss / len(train_data)
        #val_loss = validate_after_epoch(model, tokenizer,loss_criterion, validation_data )

        #print(f"[epoch: {epoch + 1}] Validation loss: {val_loss}")
        print(f"[epoch: {epoch + 1}] Training loss: {loss}")

        if loss > previous_loss:
            no_improvement_counter += 1
            previous_loss = loss
        else: no_improvement_counter = 0

        mean_loss += loss
    print(f"Average loss: {mean_loss/epochs}")
    return model

def early_stop():
    print("Implement me!")


def perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler):
    running_loss = 0.0
    for i in range(len(train_data)):
        data_sample = train_data[i]

        images, labels, text = data_sample[1].to(device), data_sample[2].to(device), tokenizer(data_sample[0],
                                                                                               return_tensors="pt",
                                                                                               is_split_into_words=True)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(images, text)
        loss = loss_criterion(outputs.squeeze(0), aligned_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        running_loss += loss.item()

        optimizer.step()
        scheduler.step()

    return running_loss

#todo consumes too much memory
def validate_after_epoch(model, tokenizer,  loss_criterion, validation_data):
    model.eval()
    loss = 0.
    for i in range(len(validation_data)):
        data_sample = validation_data[i]
        images, labels, text = data_sample[1].to(device), data_sample[2].to(device), tokenizer(data_sample[0],
                                                                                               return_tensors="pt",
                                                                                               is_split_into_words=True)
        labels = torch.tensor(align_labels(text.word_ids(),labels), device=device)
        text = {key: value.to(device) for key, value in text.items()}

        outputs = model(images, text)
        loss += loss_criterion(outputs.squeeze(0), labels)

    return loss / len(validation_data)


def inference_loop_combined_model(model: torch.nn.Module, test_data, tokenizer):
    model.eval()
    y_predicted = []
    y_actual = []
    for i in range(len(test_data)):
        sample = test_data[i]
        images = sample[1]
        labels = sample[2]
        text = tokenizer(" ".join(sample[0]), return_tensors="pt", max_length=len(labels) + 1,
                         pad_to_max_length=True)
        outputs = model(images, text)
        y_pred = outputs.view(-1, 9)[1:]
        y_predicted.append(torch.argmax(y_pred, dim=1))
        y_actual.append(labels)

    return model
