import torch
class VisualModelClassifier(torch.nn.Module):
    def __init__(self, visual_model, num_labels):
        super().__init__()
        self.visual_model = visual_model
        self.out_layer = torch.nn.Linear(visual_model.output_size, num_labels)
        self.non_linear_layer = torch.nn.Sequential(
            torch.nn.Linear(visual_model.output_size, visual_model.output_size * 2),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(visual_model.output_size * 2, visual_model.output_size)
        )

    def forward(self, visual_feats):
        visual_out = self.visual_model(visual_feats)
        visual_out = self.non_linear_layer(visual_out)
        return self.out_layer(visual_out)