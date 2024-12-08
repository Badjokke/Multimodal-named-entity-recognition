from transformers import TokenClassificationPipeline

class TokenPipeline(TokenClassificationPipeline):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer)
