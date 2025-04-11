import torch
import torch.nn.functional as F

class PartialPredictionMultimodalModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.out_layer = torch.nn.Sequential(
            torch.nn.LayerNorm(num_labels * 2),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(num_labels * 2, num_labels)
        )
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.visual_model.output_size, self.visual_model.output_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.visual_model.output_size // 2, num_labels)
        )
        self.text_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(self.text_model.config.hidden_size // 2, num_labels)
        )

    def forward(self, visual_feats, text_feats, expand=True):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state if expand else self.text_model(text_feats)

        visual_out = self.projection_layer(visual_out)
        text_out = self.text_projection_layer(text_out)

        text_out = text_out.repeat(visual_feats.size(0), 1, 1)
        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)

        return self.out_layer(torch.cat([F.softmax(visual_out, dim=-1), F.softmax(text_out, dim=-1)], dim=-1))
