from TorchCRF import CRF
import torch
class BertCRF(torch.nn.Module):
    def __init__(self,text_model, num_labels):
        super(BertCRF, self).__init__()
        self.text_model = text_model
        self.out_layer = torch.nn.Linear(text_model.config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, text_features, labels = None):
        mask = text_features["attention_mask"].bool()
        text_out = self.text_model(**text_features)
        emissions = self.out_layer(text_out)
        if labels is not None:
            loss = -self.crf(emissions[0:, 1:], labels[0:, 1:], mask=mask[0:, 1:], reduction="mean")
            predictions = self.crf.decode(emissions)
            return predictions, loss

        return self.crf.decode(emissions, mask=mask)


