from typing import Union
import torch
class LstmTokenizer:
    def __init__(self, vocabulary: dict[str, int]):
        self.vocabulary = vocabulary

    def __call__(self, *args, **kwargs):
        return self.word_to_vocab_index(args[0])

    def word_to_vocab_index(self, sentence: Union[list[str],tuple[str]]) -> torch.Tensor:
        return torch.tensor(list(map(lambda word: self.vocabulary[word], sentence)))
