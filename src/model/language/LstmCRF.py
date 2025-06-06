import torch
from torchcrf import CRF


class LstmCRF(torch.nn.Module):
    def __init__(self, text_model, num_labels):
        super().__init__()
        self.text_model = text_model
        self.out_layer = torch.nn.Linear(text_model.config.hidden_size // 2, num_labels)
        self.reduction_layer = torch.nn.Sequential(
            torch.nn.Linear(text_model.config.hidden_size, text_model.config.hidden_size // 2),
            torch.nn.Dropout(0.3),
            torch.nn.GELU(),
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, text_feats):
        text_out = self.text_model(text_feats)
        text_out = self.reduction_layer(text_out)
        return self.out_layer(text_out)

    def crf_pass(self, x, y, mask, weight):
        base_loss = -self.crf(x, y, mask, reduction="mean")
        weight_mask = weight[y]
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

    def crf_decode(self, x, mask):
        return self.crf.decode(x, mask)
