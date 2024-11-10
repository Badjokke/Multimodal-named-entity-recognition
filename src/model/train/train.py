from typing import Iterable

import torch


def _create_cross_entropy_loss_criterion() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()


def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=0.01, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)


def training_loop(model: torch.nn.Module, train_data, tokenizer, epochs=5):
    loss_criterion = _create_cross_entropy_loss_criterion()
    optimizer = _create_optimizer(model.parameters())
    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(train_data)):
            optimizer.zero_grad()

            data_sample = train_data[i]
            text = tokenizer("".join(data_sample[0]), return_tensors="pt")
            images = data_sample[1]
            labels = data_sample[2]
            outputs = model(images, text)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 500 == 499:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    return model

def perform_inference(model: torch.nn.Module, test_data):
    pass
