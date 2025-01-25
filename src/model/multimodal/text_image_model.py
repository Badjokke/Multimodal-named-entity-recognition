import torch
import torch.nn.functional as F

class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.__out_layer = torch.nn.Linear(256, num_labels)
        self.__visual_to_text_projection = torch.nn.Linear(visual_model.output_size, text_model.config.hidden_size)
        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * text_model.config.hidden_size, 856),
            torch.nn.GELU(),
            torch.nn.LayerNorm(856),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(856, 256)
        )

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state
        visual_out = self.__visual_to_text_projection(visual_out)

        visual_out = F.normalize(visual_out, p=2, dim=-1)
        text_out = F.normalize(text_out, p=2, dim=-1)
        text_out = text_out.repeat(visual_out.size(0),1,1)
        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)
        combined = torch.cat([text_out, visual_out], dim=-1)
        fused = self.__fusion_layer(combined)
        return self.__out_layer(fused)
