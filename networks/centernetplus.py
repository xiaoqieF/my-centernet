import torch
import torch.nn as nn
from networks.resnet import resnet18, resnet50_Head, resnet50
from networks.CSPDarknet import CSPDarknet, SPPF
from networks.modules import Conv, ResizeConv
from networks.smallbackbone import MobileNetv2

class CenterNetPlus(nn.Module):
    def __init__(self, num_classes, backbone="r18", pretrained=True):
        super(CenterNetPlus, self).__init__()
        self.num_classes = num_classes
        if backbone == "r18":
            self.backbone = resnet18(pretrained=pretrained)
            c2, c3, c4, c5 = 64, 128, 256, 512
            p2, p3, p4, p5 = 128, 128, 128, 128
            act = 'relu'
        elif backbone == "csp_s":
            self.backbone = CSPDarknet(base_channels=32, base_depth=1, phi='s', pretrained=pretrained)
            c2, c3, c4, c5 = 64, 128, 256, 512
            p2, p3, p4, p5 = 128, 128, 128, 128
            act = 'relu'
        elif backbone == "r50":
            self.backbone = resnet50(pretrained=pretrained)
            c2, c3, c4, c5 = 256, 512, 1024, 2048
            p2, p3, p4, p5 = 128, 128, 128, 128
            act = 'relu'
        elif backbone == "mobile":
            self.backbone = MobileNetv2(pretrained=pretrained)
            c2, c3, c4, c5 = 24, 32, 96, 1280
            p2, p3, p4, p5 = 128, 128, 128, 128
            act = 'relu'
        else:
            raise ValueError("Undefined backbone!!")
        

        # neck
        # # dilate encoder
        self.neck = SPPF(c1=c5, c2=p5)

        # upsample
        self.deconv4 = ResizeConv(c1=p5, c2=p4, act=act, scale_factor=2) # 32 -> 16
        self.latter4 = Conv(c4, p4, k=1, act=None)
        self.smooth4 = Conv(p4, p4, k=3, p=1, act=act)

        self.deconv3 = ResizeConv(c1=p4, c2=p3, act=act, scale_factor=2) # 16 -> 8
        self.latter3 = Conv(c3, p3, k=1, act=None)
        self.smooth3 = Conv(p3, p3, k=3, p=1, act=act)

        self.deconv2 = ResizeConv(c1=p3, c2=p2, act=act, scale_factor=2) #  8 -> 4
        self.latter2 = Conv(c2, p2, k=1, act=None)
        self.smooth2 = Conv(p2, p2, k=3, p=1, act=act)

        self.head = resnet50_Head(num_classes=num_classes, in_channel=128)
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        # backbone
        c2, c3, c4, c5 = self.backbone(x)

        # bottom-up
        p5 = self.neck(c5)
        p4 = self.smooth4(self.latter4(c4) + self.deconv4(p5))
        p3 = self.smooth3(self.latter3(c3) + self.deconv3(p4))
        p2 = self.smooth2(self.latter2(c2) + self.deconv2(p3))

        # detection head
        return self.head(p2)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    model = CenterNetPlus(2)
    device = torch.device("cuda:0")
    input = torch.randn([4, 3, 512, 512])
    output = model(input)
    print(output.shape)