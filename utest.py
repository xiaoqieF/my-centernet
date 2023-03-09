from utils.utils import model_info
from networks.centernet import CenterNet
from networks.centernetplus import CenterNetPlus
from networks.resnet import resnet18, resnet34, resnet50
from networks.CSPDarknet import CSPDarknet
import torch
from torchsummary import summary
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from collections import OrderedDict
import functools
import cv2
from PIL import Image
import numpy as np


if __name__ == "__main__":
    sample_img = "./samples/imgs/333.jpg"
    a = cv2.imread(sample_img)
    a = a.transpose((2, 0, 1))[::-1]
    b = Image.open(sample_img)
    b_1 = np.transpose(np.array(b, dtype=np.float32), (2, 0, 1))
    print(a)
    print(b_1)