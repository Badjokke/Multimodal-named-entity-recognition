import torch.nn as nn
import torch.nn.functional as F


class TransformerClassifier(nn.Module):
    def __init__(self, label_count, transformer_model):
        super(TransformerClassifier, self).__init__()
        self.out_layer = nn.Linear(transformer_model.config.hidden_size, label_count)
        self.transformer_model = transformer_model

    def forward(self, text_feats):
        x = self.transformer_model(**text_feats)
        return F.relu(self.out_layer(x))
