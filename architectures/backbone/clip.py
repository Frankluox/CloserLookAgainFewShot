import clip
import torch.nn as nn

class Clip(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model, _ = clip.load(name)
        self.model = self.model.float()

    def forward(self, x):
        return self.model.encode_image(x)


def create_model(name="ViT-B/16"):
    return Clip(name)



    

