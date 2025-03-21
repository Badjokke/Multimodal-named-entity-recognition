import torch
import torch.nn.functional as F
from torchcrf import CRF


class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model

        self.__out_layer = torch.nn.Linear(text_model.config.hidden_size, num_labels)

        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size // 2  + self.visual_model.output_size // 2,
                            text_model.config.hidden_size),
            torch.nn.LayerNorm(text_model.config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
        )

        self.__projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.visual_model.output_size, self.visual_model.output_size//2),
            #torch.nn.LayerNorm(self.visual_model.output_size//2),
            torch.nn.GELU(),
        )
        self.__text_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size // 2),
            #torch.nn.LayerNorm(self.text_model.config.hidden_size // 2),
            torch.nn.GELU(),
        )
        self.bilstm = torch.nn.LSTM(
            text_model.config.hidden_size,
            text_model.config.hidden_size // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.dropout = torch.nn.Dropout(0.3)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state  # (sentence_len, hidden_size)

        visual_out = self.__projection_layer(visual_out)
        text_out = self.__text_projection_layer(text_out)
        text_out = text_out.repeat(visual_feats.size(0), 1, 1)
        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)

        combined = self.__fusion_layer(torch.cat([visual_out, text_out], dim=-1))
        combined, _ = self.bilstm(combined)
        return self.__out_layer(self.dropout(combined))

    def crf_pass(self, x, y, mask, weight):
        #return -self.crf(x, y, mask, reduction="token_mean")
        base_loss = -self.crf(x, y, mask)
        weight_mask = weight[y]
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()
        """
        weight_mask = weight[y]

        weighted_loss = base_loss * weight_mask
        return weighted_loss
        """
    def crf_decode(self, x, mask):
        return self.crf.decode(x, mask)

    def __expert_pass(self, x: torch.Tensor, experts: torch.nn.ModuleList, gate: torch.nn.Module):
        router_out, indexes = gate(x)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = router_out.view(-1, router_out.size(-1))
        for i, expert in enumerate(experts):
            expert_mask = (indexes == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output

    def __compute_focal_weights(self, emissions, y, gamma):
        batch_size = emissions.size()[0]
        seq_len = emissions.size()[1]
        tag_probs = (1 - F.softmax(emissions, dim=1)) ** gamma
        correct_tag_probs = torch.zeros(batch_size, seq_len, device=emissions.device)
        for b in range(batch_size):
            for t in range(seq_len):
                if y[b, t] != -100:
                    correct_tag_probs[b, t] = tag_probs[b, t, y[b, t]]
        return correct_tag_probs
