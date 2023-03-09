import torch.nn as nn
from .resnet import resnet50, resnet18, resnet50_Decoder, resnet50_Head
from .CSPDarknet import CSPDarknet
from .darknet import darknet53
import torch.nn.functional as F
import torch


class CenterNet(nn.Module):
    def __init__(self, num_classes, backbone="r18", pretrained=False):
        super(CenterNet, self).__init__()
        # 512,512,3 -> 16,16,N
        if backbone == "r18":
            self.backbone = resnet18(pretrained=pretrained)
        elif backbone == "r50":
            self.backbone = resnet50(pretrained=pretrained)
        elif backbone == "dark53":
            self.backbone = darknet53(pretrained=pretrained)
        else:
            raise ValueError("unknow backbone")
        # 16,16,N -> 128,128,64
        self.decoder = resnet50_Decoder(self.backbone.out_channels)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(in_channel=64, num_classes=num_classes)
        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)
        
    def forward(self, x):
        _1, _2, _3, feat = self.backbone(x)
        return self.head(self.decoder(feat))

if __name__ == '__main__':
    import torch
    model = CenterNet()
    input = torch.randn((4, 3, 512, 512))
    output = model(input)