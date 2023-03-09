import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision


class MobileNetv2(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = torchvision.models.mobilenet_v2(pretrained=pretrained)
        return_layer = {"features.3": "0",
                        "features.6": "1",
                        "features.13": "2",
                        "features.18": '3'}
        self.mm = create_feature_extractor(m, return_layer)

    def forward(self, x):
        out = self.mm(x)
        return list(out.values())

class MobileNetv3(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        m = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        print(m)
        return_layer = {
            "features.3": "0",
            "features.6": "1",
            "features.12": "2",
            "features.16": "3",
        }
        
        self.mm = create_feature_extractor(m, return_layer)

    def forward(self, x):
        out = self.mm(x)
        return list(out.values())

if __name__ == "__main__":
    model = MobileNetv3(pretrained=False)
    device = torch.device("cuda:0")
    model.to(device)
    out = model(torch.randn((1, 3, 512, 512), device=device))
    print(out)