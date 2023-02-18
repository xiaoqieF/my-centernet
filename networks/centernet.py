import torch.nn as nn
from .resnet import resnet50, resnet50_Decoder, resnet50_Head
from .darknet import darknet53
import math

class CenterNet(nn.Module):
    def __init__(self, num_classes, backbone):
        super(CenterNet, self).__init__()
        # 512,512,3 -> 16,16,N
        self.backbone = backbone
        # 16,16,N -> 128,128,64
        self.decoder = resnet50_Decoder(self.backbone.out_channels)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = resnet50_Head(channel=64, num_classes=num_classes)
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
        feat = self.backbone(x)
        return self.head(self.decoder(feat))

def centernet_resnet50(num_classes=20, backbone_weight_path=""):
    return CenterNet(num_classes=num_classes, backbone=resnet50(weight_path=backbone_weight_path))

def centernet_darknet53(num_classes=20, backbone_weight_path=""):
    return CenterNet(num_classes=num_classes, backbone=darknet53(weight_path=backbone_weight_path))

if __name__ == '__main__':
    import torch
    model = CenterNet()
    input = torch.randn((4, 3, 512, 512))
    output = model(input)