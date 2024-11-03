import torch
import torch.nn.functional as F


class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        print(f"context size: {text_model.config.hidden_size}")
        self.__visual_model = visual_model
        self.__text_model = text_model
        # keep 60% of the information - llama hidden_size should be 4096
        #self.__visual_reduction = torch.nn.Linear(1080, 756)
        #self.__text_reduction = torch.nn.Linear(text_model.config.hidden_size, int(text_model.config.hidden_size * 0.7))
        self.__connector = torch.nn.Linear(1080 + text_model.config.hidden_size, 2911)
        self.__dropout = torch.nn.Dropout(0.5)
        self.__out_layer = torch.nn.Linear(2911, num_labels)

    def forward(self, visual_feats, text_feats):
        visual_out = self.__visual_model(visual_feats)
        text_out = self.__text_model(**text_feats).last_hidden_state[:, 0, :]
        combined = torch.cat((visual_out,text_out), dim=1)
        x = F.relu(self.__connector(combined))
        output = self.__out_layer(self.__dropout(x))
        return output