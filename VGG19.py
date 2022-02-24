import torch
import torch.nn as nn

from torchvision.models import vgg19

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()

        self.device = device

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.norm_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self._model = vgg19(pretrained=True).features.to(device).eval()

    def forward(self, x):
        return self._model(x)

def _load_device() -> torch.device :
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model() -> nn.Module :
    return Model(_load_device())
