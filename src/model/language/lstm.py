import torch.nn as nn
import torch.nn.functional as F
class LSTM(nn.Module):
    def __init__(self, vocab_size, bidirectional):
        super(LSTM, self).__init__()
        self.config = self.Configuration(vocab_size, 512, 300, bidirectional)
        self.embedding = nn.Embedding(vocab_size, 300) # (vocab_size, 300)
        self.lstm = nn.LSTM(300, 512, bidirectional=bidirectional)

    class Configuration:
        def __init__(self, vocab_size, hidden_size, embedding_size, bidirectional):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size if not bidirectional else hidden_size * 2
            self.embedding_size = embedding_size
            self.bidirectional = bidirectional



    def forward(self, text_features):
        x = F.relu(self.embedding(text_features))
        x, (hidden, cell) = self.lstm(x)
        x = F.relu(x)
        return x
