import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, vocab_size, bidirectional):
        super(LSTM, self).__init__()
        self.hidden_size = 256
        self.embedding_size = 300
        self.config = self.Configuration(vocab_size, self.hidden_size, self.embedding_size, bidirectional)
        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=None) # (vocab_size, 300)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers=2, dropout=0.4, bidirectional=bidirectional, batch_first=True, )
        self.layer_norm = nn.LayerNorm(self.hidden_size * 2)
        self.dropout = nn.Dropout(0.3)

    class Configuration:
        def __init__(self, vocab_size, hidden_size, embedding_size, bidirectional):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
            self.embedding_size = embedding_size
            self.bidirectional = bidirectional



    def forward(self, text_features):
        x = self.embedding(text_features)
        x, (hidden, cell) = self.lstm(x)
        x = self.dropout(self.layer_norm(x))
        return x
