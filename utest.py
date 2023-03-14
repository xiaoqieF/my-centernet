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
import time
from utils.dataset import LoadStream, LoadVideo


if __name__ == "__main__":
    video_path = './samples/111.mp4'
    video_stream = LoadVideo(video_path)
    for frame_idx, (path, img, im0s) in enumerate(video_stream):
        cv2.imshow('video', img)
        print(video_stream.frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
        time.sleep(0.01)
        img = torch.from_numpy(img)
        img = img.float() / 255
        img = img.unsqueeze(0)