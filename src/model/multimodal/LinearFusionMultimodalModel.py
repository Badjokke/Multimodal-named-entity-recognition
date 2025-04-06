import torch
from torchcrf import CRF


class LinearFusionMultimodalModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.out_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(text_model.config.hidden_size),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(text_model.config.hidden_size, num_labels)
        )

        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size // 2 + self.visual_model.output_size // 2,
                            text_model.config.hidden_size),
            torch.nn.LayerNorm(text_model.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
        )
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.visual_model.output_size, self.visual_model.output_size // 2),
            torch.nn.GELU(),
        )
        self.text_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size // 2),
            torch.nn.GELU(),
        )
        self.bilstm = torch.nn.LSTM(
            text_model.config.hidden_size,
            text_model.config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state  # (sentence_len, hidden_size)

        visual_out = self.projection_layer(visual_out)
        text_out = self.text_projection_layer(text_out)
        text_out = text_out.repeat(visual_feats.size(0), 1, 1)

        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)

        combined = self.fusion_layer(torch.cat([visual_out, text_out], dim=-1))
        combined, _ = self.bilstm(combined)
        return self.out_layer(combined)

    def crf_pass(self, x, y, mask, weight):
        base_loss = -self.crf(x, y, mask, reduction="mean")
        weight_mask = weight[y]
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

    def crf_decode(self, x, mask):
        return self.crf.decode(x, mask)