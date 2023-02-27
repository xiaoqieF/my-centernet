import torch.nn as nn
from .resnet import resnet50, resnet18, resnet50_Decoder, resnet50_Head
from .CSPDarknet import CSPDarknet
from .darknet import darknet53
import torch.nn.functional as F
import torch


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

def centernet_resnet50(num_classes=20, pretrained=False):
    return CenterNet(num_classes=num_classes, backbone=resnet50(pretrained=pretrained))

def centernet_darknet53(num_classes=20, pretrained=False):
    return CenterNet(num_classes=num_classes, backbone=darknet53(pretrained=pretrained))

def centernet_resnet18(num_classes=20, pretrained=False):
    return CenterNet(num_classes=num_classes, backbone=resnet18(pretrained=pretrained))

def centernet_yolos(num_classes=20, pretrained=True):
    depth_dict          = {'n': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
    width_dict          = {'n': 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
    dep_mul, wid_mul    = depth_dict['s'], width_dict['s']

    base_channels       = int(wid_mul * 64)  # 64
    base_depth          = max(round(dep_mul * 3), 1)  # 3
    #-----------------------------------------------#
    #   输入图片是640, 640, 3
    #   初始的基本通道是64
    #-----------------------------------------------#
    backbone       = CSPDarknet(base_channels, base_depth, 's', pretrained=pretrained)
    backbone.out_channels = 512
    return CenterNet(num_classes=num_classes, backbone=backbone)

if __name__ == '__main__':
    import torch
    model = CenterNet()
    input = torch.randn((4, 3, 512, 512))
    output = model(input)