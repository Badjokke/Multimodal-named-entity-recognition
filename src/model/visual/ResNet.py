import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ResNet(torch.nn.Module):
    def __init__(self, model, processor):
        super(ResNet, self).__init__()
        self.model = model
        self.processor = processor
        self.headless_layer = torch.nn.Sequential(*list(model.children())[:-1])
        self.output_size = 2048

    def forward(self, image_feats):
        processed = self.processor(image_feats, return_tensors='pt').to(device)
        model_out = self.headless_layer(**processed)
        #(batch_size, hidden_state_size)
        return model_out.logits