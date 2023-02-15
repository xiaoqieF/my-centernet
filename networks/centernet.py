import torch.nn as nn
from .resnet import resnet50, resnet50_Decoder, resnet50_Head
import math

class CenterNet(nn.Module):
    def __init__(self, num_classes = 20, backbone_weight=""):
        super(CenterNet, self).__init__()
        self.backbone_weight = backbone_weight
        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(weight_path=self.backbone_weight)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
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
        if self.backbone_weight == "":
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


if __name__ == '__main__':
    import torch
    model = CenterNet()
    input = torch.randn((4, 3, 512, 512))
    output = model(input)