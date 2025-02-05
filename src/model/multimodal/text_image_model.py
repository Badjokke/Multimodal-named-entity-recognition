import torch
from model.multimodal.Expert import Expert
from model.multimodal.NoisyTopkRouter import NoisyTopkRouter
import torch.nn.functional as F
from TorchCRF import CRF
class CombinedModel(torch.nn.Module):
    def __init__(self, visual_model, text_model, num_labels, num_experts=5):
        super().__init__()
        self.visual_model = visual_model
        self.text_model = text_model
        self.n_embeds = 1024

        self.__out_layer = torch.nn.Linear(768, num_labels)

        #self.__text_lin_layer = torch.nn.Linear(text_model.config.hidden_size, self.n_embeds)
        #self.text_experts = torch.nn.ModuleList([Expert(self.n_embeds) for _ in range(num_experts)])
        #self.visual_experts = torch.nn.ModuleList([Expert(visual_model.output_size) for _ in range(num_experts)])
        #self.text_gate = NoisyTopkRouter(self.n_embeds, num_experts, 3)
        #self.visual_gate = NoisyTopkRouter(visual_model.output_size, num_experts, 3)

        self.__fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(visual_model.output_size + self.n_embeds, 768),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
        )
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, visual_feats, text_feats):
        visual_out = self.visual_model(visual_feats)
        text_out = self.text_model(**text_feats).last_hidden_state  #  (sentence_len, hidden_size)

        visual_out = F.normalize(visual_out, p=2, dim=-1)
        text_out = F.normalize(text_out, p=2, dim=-1)

        text_out = text_out.repeat(visual_out.size(0), 1, 1)
        
        #text_out = self.__text_lin_layer(text_out)
        #text_out = self.__expert_pass(text_out, self.text_experts, self.text_gate)
        #visual_out = self.__expert_pass(visual_out, self.visual_experts, self.visual_gate)

        visual_out = visual_out.unsqueeze(1).expand(-1, text_out.size(1), -1)
        combined = self.__fusion_layer(torch.cat([text_out, visual_out], dim=-1))

        x = self.__out_layer(combined)
        return x

    def crf_pass(self, x,y, mask):
        return -self.crf(x, y, mask)

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