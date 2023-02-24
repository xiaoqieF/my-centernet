import torch
import torch.nn as nn
from networks.resnet import resnet18
from networks.modules import Conv, ResizeConv, DilateEncoder

class CenterNetPlus(nn.Module):
    def __init__(self, num_classes, weight_path=""):
        super(CenterNetPlus, self).__init__()
        self.num_classes = num_classes
        self.backbone = resnet18(weight_path=weight_path) 
        c2, c3, c4, c5 = 64, 128, 256, 512
        p2, p3, p4, p5 = 256, 256, 256, 256
        act = 'relu'

        # neck
        # # dilate encoder
        self.neck = DilateEncoder(c1=c5, c2=p5, act=act)

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

        # detection head
        self.cls_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, 2, kernel_size=1)
        )
       
        self.twth_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # init weight of cls_pred
        # init_prob = 0.01
        # bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # nn.init.constant_(self.cls_pred[-1].bias, bias_value)

    def forward(self, x):
        # backbone
        c2, c3, c4, c5 = self.backbone(x)
        B = c5.size(0)

        # bottom-up
        p5 = self.neck(c5)
        p4 = self.smooth4(self.latter4(c4) + self.deconv4(p5))
        p3 = self.smooth3(self.latter3(c3) + self.deconv3(p4))
        p2 = self.smooth2(self.latter2(c2) + self.deconv2(p3))

        # detection head
        cls_pred = self.cls_pred(p2)
        twth_pred = self.twth_pred(p2)
        txty_pred = self.txty_pred(p2)

        return cls_pred, twth_pred, txty_pred

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