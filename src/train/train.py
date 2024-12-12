from typing import Iterable, Union

import torch
from peft import PeftModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_cross_entropy_loss_criterion(labels) -> torch.nn.CrossEntropyLoss:
    #y = np.array(labels)
    weight = [0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]#compute_class_weight('balanced', np.unique(y),y)
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(weight, device=device), ignore_index=-100, reduction='mean')


def align_labels(word_ids, labels):
    padded_labels = []
    for word in word_ids:
        padded_labels.append(-100 if word is None else labels[word])
    return padded_labels

def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=0.01)

def _create_adamw_optimizer(parameters: Iterable[torch.Tensor], learning_rate=5e-3) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate)

def _create_scheduler(optimizer, t_max) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def training_loop_combined(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer, epochs=10, patience=3):
    model.train()
    model.to(device)

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
    model.train()
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    loss_criterion = _create_cross_entropy_loss_criterion(None)
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

    print(f"accuracy: {_calculate_accuracy(y_actual, y_predicted)}")
    print(f"test loss: {_calculate_loss(loss_criterion, y_predicted, y_actual)}")
    return model


def _calculate_loss(criterion: torch.nn.CrossEntropyLoss, y_pred: list[torch.tensor],
                    y_true: list[torch.tensor]) -> float:
    running_loss = 0.0
    for i in range(len(y_true)):
        loss = criterion(y_pred[i] / 255, y_true[i] / 255)
        running_loss += loss.item()
    return running_loss / len(y_true)


def _calculate_accuracy(y_pred: list[torch.tensor], y_true: list[torch.tensor]) -> float:
    hits = 0
    for i in range(len(y_pred)):
        if torch.eq(y_pred[i], y_true[i]).all():
            hits += 1
    return hits / len(y_pred)
