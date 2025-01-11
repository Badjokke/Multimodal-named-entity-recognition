import torch
import torch.nn.functional as F

class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.__connector = torch.nn.Linear(visual_model.output_size + text_model.config.hidden_size, 512)
        self.__out_layer = torch.nn.Linear(112, num_labels)
        self.__linear = torch.nn.Linear(512, 112)
        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(visual_model.output_size + text_model.config.hidden_size, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.3)
        )

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        #text_out = self.text_model(**text_feats).last_hidden_state  #  (sentence_len, hidden_size)
        text_out = self.text_model(text_feats).unsqueeze(0)
        visual_out = visual_out.unsqueeze(0).expand(1, text_out.size(1), -1)
        combined = torch.cat([text_out, visual_out], dim=-1)
        fused = self.__fusion_layer(combined)
        x = F.relu(self.__linear(fused))
        x = self.__out_layer(x)
        return x