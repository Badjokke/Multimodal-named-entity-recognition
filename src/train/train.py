from typing import Union

import torch
from peft import PeftModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from metrics.metrics import Metrics
from train.optimizer.OptimizerFactory import OptimizerFactory
from train.util.EarlyStop import StepState
from train.util.TrainingUtil import TrainingUtil
"""
==DISCLAIMER==
The code bellow is very bloaty and redundant due to minor differences in LSTM and Transformer implementation
that would enforce relatively long if branches I decided to copy-paste and keep the functions more readable
although way more verbose
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
-- image classifiers
"""
def image_only_training(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, test_data,
                           class_occurrences, labels, epochs=10, patience=3, learning_rates={}):
    model.to(device)
    optimizer = OptimizerFactory.create_adamw_optimizer(model, learning_rates=learning_rates)
    scheduler = OptimizerFactory.create_plateau_scheduler(optimizer)
    loss_criterion = TrainingUtil.create_cross_entropy_loss_criterion(class_occurrences)
    max_early_stop = TrainingUtil.create_maximizing_early_stop(patience=patience)
    best_state_dict = None
    training_results = []
    for epoch in range(epochs):
        training_loss = perform_epoch_image_only(model, train_data, optimizer,
                                      {value: key for key, value in labels.items()}, loss_criterion)
        val_loss = validate_after_epoch_image_only(model, validation_data,
                                        {value: key for key, value in labels.items()}, loss_criterion)
        test_results = validate_after_epoch_image_only(model,test_data, {value: key for key, value in labels.items()},loss_criterion)
        scheduler.step(val_loss[1]['macro'])
        print("---")
        print(
            f"[epoch: {epoch + 1}] Training loss: {training_loss[0]}. Training macro f1: {training_loss[1]['macro']}; micro f1: {training_loss[1]['micro']}, acc: {training_loss[1]['accuracy']}")
        print(
            f"[epoch: {epoch + 1}] Validation loss: {val_loss[0]}. Validation macro f1: {val_loss[1]['macro']}; micro f1: {val_loss[1]['micro']}, acc: {val_loss[1]['accuracy']}")
        print(
            f"[epoch: {epoch + 1}] Test loss: {test_results[0]}. Test macro f1: {test_results[1]['macro']}; micro f1: {test_results[1]['micro']}, acc: {test_results[1]['accuracy']}")
        print("---")
        training_results.append({"train": training_loss, "val": val_loss, "test": test_results})

        state = max_early_stop.verify(val_loss[1]['macro'])
        if state == StepState.STOP:
            print("Early stopping")
            break
        if state == StepState.BETTER:
            best_state_dict = model.state_dict()
    return model, training_results, best_state_dict


def perform_epoch_image_only(model, train_data, optimizer, labels_mapping, loss_criterion, scheduler=None):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []
    for i in range(len(train_data)):
        data_sample = train_data[i]
        word_count = len(data_sample[0])
        images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long, device=device)
        optimizer.zero_grad()
        labels = labels.unsqueeze(0).repeat(images.size(0), 1)
        outputs = model(images)
        outputs = outputs.unsqueeze(1).expand(-1,word_count,-1)
        loss = loss_criterion(outputs.permute(0,2,1), labels)
        running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        y_true.append(labels.tolist())
        y_pred.append(torch.argmax(outputs, dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()
    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}


def validate_after_epoch_image_only(model, train_data, labels_mapping, loss_criterion):
    model.eval()
    running_loss = 0.0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i in range(len(train_data)):
            data_sample = train_data[i]
            word_count = len(data_sample[0])
            images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long, device=device)
            labels = labels.unsqueeze(0).repeat(images.size(0), 1)
            outputs = model(images)
            outputs = outputs.unsqueeze(1).expand(-1, word_count, -1)
            loss = loss_criterion(outputs.permute(0, 2, 1), labels)
            running_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            y_true.append(labels.tolist())
            y_pred.append(torch.argmax(outputs, dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()
    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}


"""
-- lstm based methods due to different tokenizer
"""
def lstm_training(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, test_data, class_occurrences, labels, epochs=50, patience=3, text_only=False, cross_loss=False, learning_rates={}):
    model.to(device)
    optimizer = OptimizerFactory.create_adamw_optimizer(model, learning_rates=learning_rates)
    scheduler = OptimizerFactory.create_plateau_scheduler(optimizer)
    w = TrainingUtil.compute_class_weights_rare_events(class_occurrences)
    max_early_stop = TrainingUtil.create_maximizing_early_stop(patience=patience)
    best_state_dict = None
    training_results = []
    loss_criterion = None if not cross_loss else TrainingUtil.create_cross_entropy_loss_criterion(class_occurrences)
    for epoch in range(epochs):
        training_loss = perform_epoch_lstm(model, train_data, optimizer, {value: key for key, value in labels.items()}, w, text_only=text_only, loss_criterion=loss_criterion)
        val_loss = validate_after_epoch_lstm(model, validation_data, {value: key for key, value in labels.items()}, w, text_only=text_only, loss_criterion=loss_criterion)
        test_results = validate_after_epoch_lstm(model, test_data, {value: key for key, value in labels.items()}, w, text_only=text_only, loss_criterion=loss_criterion)
        scheduler.step(val_loss[1]['macro'])
        print("---")
        print(
            f"[epoch: {epoch + 1}] Training loss: {training_loss[0]}. Training macro f1: {training_loss[1]['macro']}; micro f1: {training_loss[1]['micro']}, acc: {training_loss[1]['accuracy']}")
        print(
            f"[epoch: {epoch + 1}] Validation loss: {val_loss[0]}. Validation macro f1: {val_loss[1]['macro']}; micro f1: {val_loss[1]['micro']}, acc: {val_loss[1]['accuracy']}")
        print(
            f"[epoch: {epoch + 1}] Test loss: {test_results[0]}. Test macro f1: {test_results[1]['macro']}; micro f1: {test_results[1]['micro']}, acc: {test_results[1]['accuracy']}")
        print("---")
        training_results.append({"train": training_loss, "val": val_loss, "test": test_results})

        state = max_early_stop.verify(val_loss[1]['macro'])
        if state == StepState.STOP:
            print("Early stopping")
            break
        if state == StepState.BETTER:
            best_state_dict = model.state_dict()
    return model, training_results, best_state_dict

def perform_epoch_lstm(model, train_data, optimizer, labels_mapping, w, scheduler=None, text_only=False, loss_criterion=None):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text = data_sample[0]
        images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,device=device)

        optimizer.zero_grad()

        aligned_labels = labels.unsqueeze(0)
        if not text_only:
            aligned_labels = aligned_labels.repeat(images.size(0), 1)
            outputs = model(images, text, unpack=False)
        else:
            outputs = model(text)
        mask = torch.ones_like(aligned_labels, device=device).bool()

        loss = model.crf_pass(outputs, aligned_labels, mask, w) if loss_criterion is None else loss_criterion(outputs.permute(0,2,1), aligned_labels)
        running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        y_true.append(aligned_labels.tolist())
        y_pred.append(model.crf_decode(outputs, mask) if loss_criterion is None else torch.argmax(outputs, dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}

def validate_after_epoch_lstm(model, validation_data, labels_mapping, w, text_only=False, loss_criterion=None) -> tuple[
    float, dict[str, float]]:
    model.eval()
    running_loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(validation_data)):
            data_sample = validation_data[i]
            text = data_sample[0]
            images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long, device=device)
            aligned_labels = labels.unsqueeze(0)

            if not text_only:
                aligned_labels = aligned_labels.repeat(images.size(0), 1)
                outputs = model(images, text, unpack=False)
            else:
                outputs = model(text)

            mask = torch.ones_like(aligned_labels, device=device).bool()
            loss = model.crf_pass(outputs, aligned_labels, mask, w) if loss_criterion is None else loss_criterion(outputs.permute(0, 2, 1), aligned_labels)

            running_loss += loss.item()
            y_true.append(aligned_labels.tolist())
            y_pred.append(model.crf_decode(outputs, mask) if loss_criterion is None else torch.argmax(outputs, dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    return running_loss / len(validation_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}

"""
-- transformer based with tokenizer
"""
def transformer_training(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, test_data, tokenizer,
                         class_occurrences, labels, epochs=10, patience=3, text_only=False, cross_loss=False, learning_rates={}, val_on_test=False):
    model.to(device)
    optimizer = OptimizerFactory.create_adamw_optimizer(model, learning_rates=learning_rates)
    scheduler = OptimizerFactory.create_plateau_scheduler(optimizer)
    w = TrainingUtil.compute_class_weights_rare_events(class_occurrences)
    max_early_stop = TrainingUtil.create_maximizing_early_stop(patience=patience)
    best_state_dict = None
    training_results = []
    loss_criterion = TrainingUtil.create_cross_entropy_loss_criterion(class_occurrences) if cross_loss else None
    for epoch in range(epochs):
        training_loss = perform_epoch(model, tokenizer, train_data, optimizer, {value: key for key, value in labels.items()}, w, text_only=text_only, loss_criterion=loss_criterion)
        val_loss = validate_after_epoch(model, tokenizer,  validation_data, {value: key for key, value in labels.items()}, w, text_only=text_only,loss_criterion=loss_criterion)
        test_results = validate_after_epoch(model, tokenizer, test_data, {value: key for key, value in labels.items()}, w, text_only=text_only,loss_criterion=loss_criterion)
        scheduler.step(val_loss[1]['macro'])
        print("---")
        print(f"[epoch: {epoch + 1}] Training loss: {training_loss[0]}. Training macro f1: {training_loss[1]['macro']}; micro f1: {training_loss[1]['micro']}, acc: {training_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Validation loss: {val_loss[0]}. Validation macro f1: {val_loss[1]['macro']}; micro f1: {val_loss[1]['micro']}, acc: {val_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Test loss: {test_results[0]}. Test macro f1: {test_results[1]['macro']}; micro f1: {test_results[1]['micro']}, acc: {test_results[1]['accuracy']}")
        print("---")
        training_results.append({"train":training_loss, "val": val_loss, "test":test_results})
        #the only reason to use val_on_test = True for early stopping is if validation set is super small which is the case for SOA
        state = max_early_stop.verify(val_loss[1]['macro']) if not val_on_test else max_early_stop.verify(test_results[1]['macro'])
        if state == StepState.STOP:
            print("Early stopping")
            break
        if state == StepState.BETTER:
            best_state_dict = model.state_dict()
    return model, training_results, best_state_dict


def perform_epoch(model, tokenizer, train_data, optimizer, labels_mapping, w, scheduler=None, text_only=False, loss_criterion=None):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []
    stripping_function = TrainingUtil.remove_cls_tokens_bert if type(tokenizer) == BertTokenizerFast else TrainingUtil.remove_cls_tokens_llama
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text = tokenizer(data_sample[0], return_tensors="pt", is_split_into_words=True).to(device) if tokenizer is not None else data_sample[0]
        images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,device=device)

        word_ids = text.word_ids()
        optimizer.zero_grad()

        aligned_labels = torch.tensor(TrainingUtil.align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0)
        if not text_only:
            aligned_labels = aligned_labels.repeat(images.size(0), 1)
            outputs = model(images, text)
        else:
            outputs = model(text)
        outputs, aligned_labels = stripping_function(outputs, aligned_labels)
        mask = torch.ones_like(aligned_labels, device=device).bool()

        loss = model.crf_pass(outputs, aligned_labels, mask, w) if loss_criterion is None else loss_criterion(outputs.permute(0,2,1), aligned_labels)
        running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        y_true.append(aligned_labels.tolist())
        y_pred.append(model.crf_decode(outputs, mask) if loss_criterion is None else torch.argmax(outputs,dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}

def validate_after_epoch(model, tokenizer, validation_data, labels_mapping, w, text_only=False, loss_criterion=None) -> tuple[
    float, dict[str, float]]:
    model.eval()
    running_loss = 0.
    y_true, y_pred = [], []
    stripping_function = TrainingUtil.remove_cls_tokens_bert if type(tokenizer) == BertTokenizerFast else TrainingUtil.remove_cls_tokens_llama
    with torch.no_grad():
        for i in range(len(validation_data)):
            data_sample = validation_data[i]
            text = tokenizer(data_sample[0], return_tensors="pt",
                             is_split_into_words=True).to(device) if tokenizer is not None else data_sample[0]
            images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long, device=device)

            word_ids = text.word_ids()

            aligned_labels = torch.tensor(TrainingUtil.align_labels(word_ids, labels), device=device,
                                          dtype=torch.long).unsqueeze(0)

            if not text_only:
                aligned_labels = aligned_labels.repeat(images.size(0), 1)
                outputs = model(images, text)
            else:
                outputs = model(text)
            outputs, aligned_labels = stripping_function(outputs, aligned_labels)
            mask = torch.ones_like(aligned_labels, device=device).bool()
            loss = model.crf_pass(outputs, aligned_labels, mask, w) if loss_criterion is None else loss_criterion(outputs.permute(0, 2, 1), aligned_labels)

            running_loss += loss.item()
            y_true.append(aligned_labels.tolist())
            y_pred.append(model.crf_decode(outputs, mask) if loss_criterion is None else torch.argmax(outputs, dim=-1).tolist())

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    return running_loss / len(validation_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}