from typing import Iterable

import torch
from sklearn.metrics import accuracy_score, f1_score


def _create_cross_entropy_loss_criterion() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()


def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-4, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


def training_loop(model: torch.nn.Module, train_data, tokenizer, epochs=1):
    loss_criterion = _create_cross_entropy_loss_criterion()
    optimizer = _create_optimizer(model.parameters())
    y_predicted = []
    y_actual = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(train_data)):
            data_sample = train_data[i]
            images = data_sample[1]
            labels = data_sample[2]
            text = tokenizer("".join(data_sample[0]), return_tensors="pt", max_length=len(labels) + 1,
                             pad_to_max_length=True)
            outputs = model(images, text)
            y_pred = outputs.view(-1, 9)[1:]
            loss = loss_criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_predicted.extend(y_pred)
            y_actual.extend(labels)

            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
        print(f"[{epoch + 1}], macro-f1: {f1_score(y_actual, y_predicted, average='macro')}")
        print(f"[{epoch + 1}], micro-f1: {f1_score(y_actual, y_predicted, average='micro')}")

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
        text = tokenizer("".join(sample[0]), return_tensors="pt", max_length=len(labels) + 1,
                         pad_to_max_length=True)
        outputs = model(images, text)
        y_pred = outputs.view(-1, 9)[1:]
        y_predicted.extend(torch.argmax(y_pred,dim=1))
        y_actual.extend(labels)
    print(f"macro-f1: {f1_score(y_actual, y_predicted, average='macro')}")
    print(f"micro-f1: {f1_score(y_actual, y_predicted, average='micro')}")
    print(f"accuracy: {accuracy_score(y_actual, y_predicted)}")
    print(f"test loss: {loss_criterion(y_predicted, y_actual)}")
    return model

