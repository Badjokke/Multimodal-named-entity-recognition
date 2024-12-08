import torch


# todo garbage model rework
class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.__visual_model = visual_model
        self.__text_model = text_model
        self.__connector = torch.nn.Linear(1080 + text_model.config.hidden_size, 2911)
        self.__dropout = torch.nn.Dropout(0.5)
        self.__out_layer = torch.nn.Linear(1024, num_labels)
        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(1080 + text_model.config.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(1024),
            torch.nn.Dropout(0.3)
        )

    def forward(self, visual_feats, text_feats):
        visual_out = self.__visual_model(visual_feats)
        text_out = self.__text_model(**text_feats).last_hidden_state  # Shape: [sentence_len, hidden_size]
        visual_out = visual_out.unsqueeze(0).expand(1, text_out.size(1), -1)
        combined = torch.cat([text_out, visual_out], dim=-1)
        fused = self.__fusion_layer(combined)
        x = self.__out_layer(fused)
        return x