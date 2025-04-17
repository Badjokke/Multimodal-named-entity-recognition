import torch.nn as nn
from model.language.LstmTokenizer import LstmTokenizer
class LSTM(nn.Module):
    def __init__(self, vocab, bidirectional):
        super(LSTM, self).__init__()
        self.vocab_size = len(vocab.keys())
        self.hidden_size = 512
        self.tokenizer = LstmTokenizer(vocab)
        self.embedding_size = 300

        self.config = self.Configuration(self.vocab_size, self.hidden_size, self.embedding_size, bidirectional)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=None) # (vocab_size, 300)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=3, dropout=0.3, bidirectional=bidirectional, batch_first=True, )
        self.layer_norm = nn.LayerNorm(self.hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

    class Configuration:
        def __init__(self, vocab_size, hidden_size, embedding_size, bidirectional):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
            self.embedding_size = embedding_size
            self.bidirectional = bidirectional

    def forward(self, text_features):
        text_features = self.tokenizer(text_features)
        #add batch dimension if input is not batched
        if len(text_features.size()) == 1:
            text_features = text_features.unsqueeze(0)
        x = self.embedding(text_features)
        x, (hidden, cell) = self.lstm(x)
        return self.dropout(self.layer_norm(x))
