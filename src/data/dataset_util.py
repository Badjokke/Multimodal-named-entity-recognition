
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=1024, padding="longest")
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words        
        label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]  # Align labels with tokens
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess_dataset_class(dataset, tokenizer):
    return dataset.map(lambda item: tokenize_and_align_labels(item, tokenizer), batched=True)
