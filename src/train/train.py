from typing import Iterable, Union

import numpy as np
import torch
from peft import PeftModel
from sklearn.utils.class_weight import compute_class_weight
from metrics.metrics import  Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _compute_inverse_freq_weights(y, alpha=0.5) -> torch.Tensor:
    weights = [1 / (np.bincount(x) ** alpha) for x in y]
    return torch.Tensor(weights).to(torch.float32).to(device)

def _compute_class_weights_rare_events(y) -> torch.Tensor:
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    return torch.from_numpy(weights).to(torch.float32).to(device)

def _create_cross_entropy_loss_criterion(y) -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss(ignore_index=-100, weight=_compute_class_weights_rare_events(y))

def align_labels(word_ids, labels):
    padded_labels = []
    for word in word_ids:
        padded_labels.append(-100 if word is None else labels[word])
    return padded_labels

def _create_optimizer(parameters: Iterable[torch.Tensor], learning_rate=1e-5, momentum=0.9) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum)

def _create_adamw_optimizer(parameters: Iterable[torch.Tensor], learning_rate=2e-5) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)

def _create_scheduler(optimizer, t_max) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, verbose=True)

def training_loop_text_only(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, tokenizer, epochs=10,
                            patience=3):
    model.to(device)
    model.train()

    loss_criterion = _create_cross_entropy_loss_criterion(None)
    optimizer = _create_optimizer(model.parameters())

    mean_loss = 0

    for epoch in range(epochs):
        loss, f1 = perform_epoch_text_only(model, tokenizer, train_data, loss_criterion, optimizer)
        loss = loss / len(train_data)
        print(f"[epoch: {epoch + 1}] Training loss: {loss}")
        mean_loss += loss
    print(f"Average loss: {mean_loss / epochs}")
    return model

def perform_epoch_text_only(model, tokenizer, train_data, loss_criterion, optimizer, scheduler):
    running_loss = 0.0
    y_pred, y_true = [], []
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text, labels = tokenizer(data_sample[0], return_tensors="pt", is_split_into_words=True), data_sample[1].to(device)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}
        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0)

        optimizer.zero_grad()

        outputs, loss = model(text,aligned_labels)
        #loss = loss_criterion(outputs.squeeze(0), aligned_labels)
        loss.backward()
        running_loss += loss.item()

        optimizer.step()
        #scheduler.step()
        y_pred.append(decode_labels_majority_vote(word_ids[1:], torch.argmax(outputs, dim=2)[0:, 1:]).tolist())
        y_true.append((labels.unsqueeze(0).repeat(outputs.size(0), 1)).tolist())

    metrics = Metrics(y_pred, y_true, 9, {_: str(_) for _ in range(9)})
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    return running_loss / len(train_data), macro_f1

def training_loop_combined(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, test_data, tokenizer,
                           class_occurrences, epochs=10, patience=3):
    model.to(device)
    loss_criterion = _create_cross_entropy_loss_criterion(class_occurrences)
    optimizer = _create_adamw_optimizer(model.parameters())
    scheduler = _create_scheduler(optimizer, t_max=epochs * len(train_data))
    #optimizer, scheduler = setup_optimizer(model, t_max=epochs*len(train_data))
    for epoch in range(epochs):
        training_loss = perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler)
        val_loss = validate_after_epoch(model, tokenizer, loss_criterion, validation_data)
        test_results = validate_after_epoch(model, tokenizer, loss_criterion, test_data)
        print(f"[epoch: {epoch + 1}] Training loss: {training_loss[0]}. Training macro f1: {training_loss[1]['macro']}; micro f1: {training_loss[1]['micro']}, acc: {training_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Validation loss: {val_loss[0]}. Validation macro f1: {val_loss[1]['macro']}; micro f1: {val_loss[1]['micro']}, acc: {val_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Test loss: {test_results[0]}. Test macro f1: {test_results[1]['macro']}; micro f1: {test_results[1]['micro']}, acc: {test_results[1]['accuracy']}")
    return model

def early_stop():
    print("Implement me!")

def perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []
    for i in range(len(train_data)):
        data_sample = train_data[i]
        images, labels, text = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,
                                                                       device=device), tokenizer(data_sample[0],
                                                                                                 return_tensors="pt",
                                                                                                 is_split_into_words=True)
        word_ids = text.word_ids()
        text = {key: value.to(device) for key, value in text.items()}

        optimizer.zero_grad()

        aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0).repeat(images.size(0),1)
        #aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0)
        outputs = model(images, text)
        aligned_labels = aligned_labels[0:, 1:]
        outputs = outputs[0:, 1:]
        mask = torch.ones_like(aligned_labels,device=device).bool()
        loss = model.crf_pass(outputs, aligned_labels, mask)

        #loss = loss_criterion(outputs.permute(0,2,1), aligned_labels)
        running_loss += loss.item()
        #y_pred.append(decode_labels_majority_vote(word_ids[1:], torch.argmax(outputs.squeeze(0),dim=1)[1:]))
        #y_true.append(labels)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        y_pred.append(model.crf_decode(outputs, mask))
        #y_pred.append(torch.argmax(outputs, dim=-1).tolist())
        y_true.append(aligned_labels.tolist())

    metrics = Metrics(y_pred, y_true, 9, {_:str(_) for _ in range(9)})
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()
    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}

def validate_after_epoch(model, tokenizer, loss_criterion, validation_data) -> tuple[
    float, dict[str,float]]:
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
            word_ids = text.word_ids()

            aligned_labels = torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0).repeat(images.size(0),1)
            text = {key: value.to(device) for key, value in text.items()}

            outputs = model(images, text)
            loss += loss_criterion(outputs.permute(0, 2, 1), aligned_labels)
            #y_pred.append(decode_labels_majority_vote(word_ids[1:], torch.argmax(outputs.squeeze(0), dim=1)[1:]))
            y_pred.append(torch.argmax(outputs, dim=-1)[0:, 1:].tolist())
            y_true.append(aligned_labels[0:, 1:].tolist())
            #y_pred.append(decode_labels_majority_vote(word_ids[1:], torch.argmax(outputs, dim=2)[0:, 1:]).tolist())
            #y_true.append((labels.unsqueeze(0).repeat(outputs.size(0), 1)).tolist())

    metrics = Metrics(y_pred, y_true, 9, {_: str(_) for _ in range(9)})
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()
    return loss / len(validation_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc} #(y_true, y_pred)


def most_common(lst):
    return max(set(lst), key=lst.count)

def decode_labels_majority_vote(word_ids, y_pred):
    y_predicted = y_pred.tolist()
    batched_decoded_labels = []
    for batch in range(len(y_predicted)):
        current_word_labels = []
        decoded_labels = []
        current_word_id = word_ids[0]
        for i in range(len(y_predicted[batch])):
            w_id = word_ids[i]
            if w_id is None:
                continue
            if current_word_id != w_id:
                decoded_labels.append(most_common(current_word_labels))
                current_word_labels = []
                current_word_id = w_id
            current_word_labels.append(y_predicted[batch][i])
        decoded_labels.append(most_common(current_word_labels))
        batched_decoded_labels.append(torch.tensor(decoded_labels, dtype=torch.long,device=device))
    return torch.stack(batched_decoded_labels)

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
