from typing import Iterable, Union
from train.util.TrainingUtil import TrainingUtil
import torch
from peft import PeftModel
#from seqeval.metrics import classification_report
from train.optimizer import OptimizerFactory
from metrics.metrics import Metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _create_cross_entropy_loss_criterion(y) -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss(ignore_index=-100, weight=TrainingUtil.compute_class_weights_rare_events(y))


def training_loop_combined(model: Union[torch.nn.Module, PeftModel], train_data, validation_data, test_data, tokenizer,
                           class_occurrences, labels, epochs=10, patience=3):
    model.to(device)
    loss_criterion = _create_cross_entropy_loss_criterion(class_occurrences)
    optimizer = OptimizerFactory.create_adamw_optimizer(model)
    scheduler = OptimizerFactory.create_plateau_scheduler(optimizer)
    w = TrainingUtil.compute_class_weights_rare_events(class_occurrences)
    training_results = []
    validate_after_epoch(model, tokenizer, loss_criterion, test_data,
                         {value: key for key, value in labels.items()}, w)
    # optimizer, scheduler = setup_optimizer(model, t_max=epochs*len(train_data))
    for epoch in range(epochs):
        #print(f"==EPOCH {epoch}==")
        # val_loss = validate_after_epoch(model, tokenizer, loss_criterion, validation_data, {value:key for key,value in labels.items()})
        #print("==TRAIN==")
        training_loss = perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler,
                                      {value: key for key, value in labels.items()}, w)
        #print("==VAL==")
        val_loss = validate_after_epoch(model, tokenizer, loss_criterion, validation_data,
                                        {value: key for key, value in labels.items()}, w)


        #print("==TEST==")
        test_results = validate_after_epoch(model, tokenizer, loss_criterion, test_data,
                                            {value: key for key, value in labels.items()}, w)
        scheduler.step(val_loss[1]['macro'])

        print(f"[epoch: {epoch + 1}] Training loss: {training_loss[0]}. Training macro f1: {training_loss[1]['macro']}; micro f1: {training_loss[1]['micro']}, acc: {training_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Validation loss: {val_loss[0]}. Validation macro f1: {val_loss[1]['macro']}; micro f1: {val_loss[1]['micro']}, acc: {val_loss[1]['accuracy']}")
        print(f"[epoch: {epoch + 1}] Test loss: {test_results[0]}. Test macro f1: {test_results[1]['macro']}; micro f1: {test_results[1]['micro']}, acc: {test_results[1]['accuracy']}")
        print()
        training_results.append({"train":training_loss, "validation": val_loss, "test":test_results})
    return model

def perform_epoch(model, tokenizer, train_data, loss_criterion, optimizer, scheduler, labels_mapping, w):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []
    for i in range(len(train_data)):
        data_sample = train_data[i]
        text = tokenizer(data_sample[0], return_tensors="pt", is_split_into_words=True) if tokenizer is not None else data_sample[0]
        images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long,device=device)

        optimizer.zero_grad()

        aligned_labels = labels.unsqueeze(0).repeat(images.size(0), 1) #torch.tensor(align_labels(word_ids, labels), device=device, dtype=torch.long).unsqueeze(0).repeat(images.size(0), 1)
        outputs = model(images, text)

        #aligned_labels = aligned_labels[0:, 1:-1]
        #outputs = outputs[0:, 1:-1]
        mask = torch.ones_like(aligned_labels, device=device).bool()

        loss = model.crf_pass(outputs, aligned_labels, mask, w)
        #loss = loss_criterion(torch.permute(outputs, (0, 2, 1)), labels)
        running_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss.backward()
        optimizer.step()
        #scheduler.step()

        #y_pred.append(decode_labels_majority_vote(word_ids[1:-1],model.crf_decode(outputs, mask)).tolist())
        #y_pred.append(torch.argmax(outputs, dim=-1).tolist())
        #y_tr_mapped = map_to_base_labels(aligned_labels.tolist(), labels_mapping)
        y_true.append(aligned_labels.tolist())
        y_pred.append(model.crf_decode(outputs, mask))

    metrics = Metrics(y_pred, y_true, len(labels_mapping.keys()), labels_mapping)
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    # print("===TRAIN===")
    # print(classification_report(y_true, y_pred))
    return running_loss / len(train_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}


def validate_after_epoch(model, tokenizer, loss_criterion, validation_data, labels_mapping, w) -> tuple[
    float, dict[str, float]]:
    model.eval()
    #model.to(device)
    running_loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(validation_data)):
            data_sample = validation_data[i]
            text = tokenizer(data_sample[0], return_tensors="pt",
                             is_split_into_words=True) if tokenizer is not None else data_sample[0]

            images, labels = data_sample[1].to(device), torch.tensor(data_sample[2], dtype=torch.long, device=device)

            #word_ids = text.word_ids()

            aligned_labels = labels.unsqueeze(0).repeat(images.size(0), 1)
            #text = {key: value.to(device) for key, value in text.items()}
            #labels = labels.repeat(images.size(0), 1)

            outputs = model(images, text)
            #aligned_labels = aligned_labels[0:, 1:-1]

            #outputs = outputs[0:, 1:-1]
            mask = torch.ones_like(aligned_labels, device=device).bool()
            loss = model.crf_pass(outputs, aligned_labels, mask, w)
            #loss = loss_criterion(torch.permute(outputs, (0, 2, 1)), labels)
            running_loss += loss.item()
            #y_pred.append(torch.argmax(outputs, dim=-1))
            y_true.append(aligned_labels.tolist())
            y_pred.append(model.crf_decode(outputs, mask))
            #y_pred.append(decode_labels_majority_vote(word_ids[1:-1],model.crf_decode(outputs, mask)))
            #y_true.append(labels.repeat(images.size(0), 1).tolist())

    metrics = Metrics(y_pred, y_true, 9, {i:i for i in range(9)})
    macro_f1 = metrics.macro_f1(metrics.confusion_matrix())
    micro_f1 = metrics.micro_f1(metrics.confusion_matrix())
    acc = metrics.accuracy()

    return running_loss / len(validation_data), {"macro": macro_f1, "micro": micro_f1, "accuracy": acc}  # (y_true, y_pred)