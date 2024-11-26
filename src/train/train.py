from typing import Iterable

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _create_cross_entropy_loss_criterion() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()


def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-4, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


def training_loop(model: torch.nn.Module, train_data, tokenizer, epochs=1):
    loss_criterion = _create_cross_entropy_loss_criterion()
    optimizer = _create_optimizer(model.parameters())
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(train_data)):
            data_sample = train_data[i]
            images, labels = data_sample[1].to(device), data_sample[2].to(device)
            text = tokenizer(" ".join(data_sample[0]), return_tensors="pt", max_length=len(labels) + 1,
                             pad_to_max_length=True)
            text = {key: value.to(device) for key, value in text.items()}
            outputs = model(images, text)
            y_pred = outputs.view(-1, 9)[1:]
            loss = loss_criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            
            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    return model


def inference_loop(model: torch.nn.Module, test_data, tokenizer):
    model.eval()
    loss_criterion = _create_cross_entropy_loss_criterion()
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
        y_predicted.append(torch.argmax(y_pred,dim=1))
        y_actual.append(labels)

    #print(f"macro-f1: {f1_score(y_actual, y_predicted, average='macro')}")
    #print(f"micro-f1: {f1_score(y_actual, y_predicted, average='micro')}")
    print(f"accuracy: {_calculate_accuracy(y_actual, y_predicted)}")
    print(f"test loss: {_calculate_loss(loss_criterion, y_predicted, y_actual)}")
    return model

def _calculate_loss(criterion: torch.nn.CrossEntropyLoss, y_pred:list[torch.tensor], y_true:list[torch.tensor]) -> float:
    running_loss = 0.0
    for i in range(len(y_true)):
        loss = criterion(y_pred[i], y_true[i])
        running_loss += loss.item()
    return running_loss / len(y_true)

def _calculate_accuracy(y_pred:list[torch.tensor], y_true:list[torch.tensor]) -> float:
    hits = 0
    for i in range(len(y_pred)):
        if torch.eq(y_pred[i], y_true[i]).all():
            hits += 1
    return hits / len(y_pred)