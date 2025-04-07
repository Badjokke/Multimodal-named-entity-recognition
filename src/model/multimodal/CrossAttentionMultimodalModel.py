import torch
from torchcrf import CRF
class CrossAttentionModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.hidden_state_size = 768
        self.text_projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.text_model.config.hidden_size, self.hidden_state_size),
        )
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(self.visual_model.output_size, self.hidden_state_size),
        )

        self.text_cross_attention = torch.nn.MultiheadAttention(
            self.hidden_state_size,
            num_heads=4,
            batch_first=True,
        )

        self.image_attention = torch.nn.MultiheadAttention(
            self.hidden_state_size,
            num_heads=4,
            batch_first=True,
        )

        self.image_cross_attention = torch.nn.MultiheadAttention(
            self.hidden_state_size,
            num_heads=4,
            batch_first=True,
        )
        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_state_size * 2,
                            self.text_model.config.hidden_size // 2),
            torch.nn.LayerNorm(self.text_model.config.hidden_size // 2),
            torch.nn.GELU()
        )
        self.__out_layer = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.text_model.config.hidden_size // 2, num_labels)
        )

        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state
        text_out = text_out.repeat(visual_feats.size(0), 1, 1)
        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)

        visual_out = self.projection_layer(visual_out)
        text_out = self.text_projection_layer(text_out)

        image_attended, _ = self.image_attention(visual_out, visual_out, visual_out)

        image_attended, _ = self.text_cross_attention(visual_out, text_out, text_out)
        text_attended, _ = self.image_cross_attention(text_out, visual_out + image_attended, visual_out + image_attended)

        combined = self.__fusion_layer(torch.cat([text_attended+text_out, image_attended], dim=-1))
        return self.__out_layer(combined)

    def crf_pass(self, x, y, mask, weight):
        base_loss = -self.crf(x, y, mask, reduction="mean")
        weight_mask = weight[y]
        weighted_loss = base_loss * weight_mask
        return weighted_loss.mean()

    def focal_weight(self, x, y, gamma=4.0):
        # Calculate prediction probabilities
        emissions = x.clone().detach()
        log_probs = torch.nn.functional.log_softmax(emissions, dim=-1)
        probs = torch.exp(log_probs)

        # Get target probabilities
        target_probs = probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)

        # Focal loss factor
        return (1 - target_probs) ** gamma

    def crf_decode(self, x, mask):
        return self.crf.decode(x, mask)
