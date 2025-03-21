import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ViT(torch.nn.Module):
    def __init__(self, model, processor):
        super(ViT, self).__init__()
        self.model = model
        self.processor = processor
        self.output_size = 768

    def forward(self, image_feats):
        processed = self.processor(image_feats, return_tensors='pt', do_rescale=False).to(device)
        model_out = self.model(**processed)
        #(batch_size, hidden_state_size)
        return model_out.pooler_output