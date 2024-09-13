from transformers import pipeline

def create_pipe(task, model, tokenizer):
    return pipeline(task=task, model= model, tokenizer= tokenizer,max_length=200)
    