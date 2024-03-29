import torch
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        in_channels: channel of input feature map
        out_channels: channel of middle feature map
        outputs channels of Bottleneck is expansion * out_channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*self.expansion, 
            kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, blocks_num):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # only first block of layer use downsample
        if stride != 1 or self.in_channels != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, channel, downsample=downsample, stride=stride))
        self.in_channels = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        C_1 = self.conv1(x)
        C_1 = self.bn1(C_1)
        C_1 = self.relu(C_1)
        C_1 = self.maxpool(C_1)

        C_2 = self.layer1(C_1)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)

        return C_2, C_3, C_4, C_5

def resnet18(pretrained=True):
    #  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    resnet_backbone = ResNet(BasicBlock, [2, 2, 2, 2])

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls["resnet18"], map_location="cpu", model_dir="./model_data")
        resnet_backbone.load_state_dict(state_dict=checkpoint, strict=False)
    resnet_backbone.out_channels = 512

    return resnet_backbone

def resnet34(pretrained=True):
    # 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    resnet_backbone = ResNet(BasicBlock, [3, 4, 6, 3])

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls["resnet34"], map_location="cpu", model_dir="./model_data")
        resnet_backbone.load_state_dict(state_dict=checkpoint, strict=False)
    resnet_backbone.out_channels = 512

    return resnet_backbone

def resnet50(pretrained=True):
    # 'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3])

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls["resnet50"], map_location="cpu", model_dir="./model_data")
        resnet_backbone.load_state_dict(state_dict=checkpoint, strict=False)
    resnet_backbone.out_channels = 2048

    return resnet_backbone

def resnet101(pretrained=True):
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    resnet_backbone = ResNet(Bottleneck, [3, 4, 23, 3])

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls["resnet101"], map_location="cpu", model_dir="./model_data")
        resnet_backbone.load_state_dict(state_dict=checkpoint, strict=False)
    resnet_backbone.out_channels = 2048

    return resnet_backbone

class resnet50_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False
        
        #----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[512, 512, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class resnet50_Head(nn.Module):
    def __init__(self, num_classes=20, in_channel=256, bn_momentum=0.1):
        super(resnet50_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channel, 64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(in_channel, 64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channel, 64,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset

if __name__ == '__main__':
    net = resnet101()
    print(net)