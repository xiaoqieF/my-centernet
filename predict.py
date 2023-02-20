import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from networks.centernet import centernet_resnet18
from utils.boxes import decode_bbox, postprocess
from utils.draw_boxes_utils import draw_box
from utils.utils import load_class_names

imgs_path = "./samples"

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

if __name__ == '__main__':
    class_names = load_class_names("./DroneYoloDataset/my_data_label.names")
    device = torch.device("cuda:0")
    model = centernet_resnet18(num_classes=1)
    model.load_state_dict(torch.load("./run/centernet_resnet18_99.pth"))
    model.to(device)

    for file in os.listdir(imgs_path):
        img = Image.open(os.path.join(imgs_path, file))
        img = resize_image(img, (512, 512), True)
        
        img = transforms.ToTensor()(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)
        output = decode_bbox(output[0], output[1], output[2], confidence=0.05)
        output = postprocess(output)[0]

        print(f"predictions: {output}")

        img = draw_box(transforms.ToPILImage()(img.squeeze(0)), output[:, :4], output[:, -1], output[:, 4], class_names)
        plt.imshow(img)
        plt.show()