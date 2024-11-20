import torch
import torch.nn.functional as F


# todo garbage model rework
class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        print(f"context size: {text_model.config.hidden_size}")
        self.__visual_model = visual_model
        self.__text_model = text_model
        self.__connector = torch.nn.Linear(1080 + text_model.config.hidden_size, 2911)
        self.__dropout = torch.nn.Dropout(0.5)
        self.__out_layer = torch.nn.Linear(2911, num_labels)

    def forward(self, visual_feats, text_feats):
        visual_out = self.__visual_model(visual_feats)
        text_out = self.__text_model(**text_feats).last_hidden_state  # Shape: [sentence_len, hidden_size]
        combined = torch.cat([visual_out.unsqueeze(0).unsqueeze(0).expand(-1, text_out.size()[1], -1), text_out], dim=-1)
        x = F.relu(self.__connector(combined))
        output = self.__out_layer(self.__dropout(x))
        return output

    def predict(self, visual_feats, text_feats):
        logits = self.forward(visual_feats, text_feats)
        probabilities = F.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probabilities, dim=2)

        return predicted_labels
