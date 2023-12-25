import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from networks.centernetplus import CenterNetPlus
from networks.centernet import CenterNet
from utils.boxes import postprocess, correct_boxes, BBoxDecoder
from utils.draw_boxes_utils import draw_box
from utils.utils import load_class_names

imgs_path = "./samples/imgs1"
video_path = "./samples/DroneVsBirds_2.mp4"
mode = "image"    # image / video

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
        new_image = Image.new('RGB', size, (144, 144, 144))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

if __name__ == '__main__':
    class_names = load_class_names("./DroneVsBirds/my_data_label.names")
    device = torch.device("cuda:0")
    model = CenterNet(num_classes=1, backbone="r18")
    model.load_state_dict(torch.load("./run/DroneVsBirds_centernet_r18_best.pth"), strict=False)
    model.to(device)
    model.eval()

    if mode == "image":
        fps = 0.0
        with torch.no_grad():
            for file in os.listdir(imgs_path):
                img_origin = Image.open(os.path.join(imgs_path, file))
                img = resize_image(img_origin, (512, 512), True)
                
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                t1 = time.time()
                output = model(img)
                print(f"time: {time.time() - t1}")
                output = BBoxDecoder.decode_bbox(output[0], output[1], output[2], confidence=0.3)
                output = postprocess(output)[0].numpy()

                output[:, 0:4] = correct_boxes(output[:, 0:4], (512, 512), img_origin.size)  # height, width
                # print(f"predictions: {output}")

                img = draw_box(img_origin, output[:, :4], output[:, -1], output[:, 4], class_names)
                plt.imshow(img)
                plt.show()
    elif mode == "video":
        cap = cv2.VideoCapture(video_path)
        size    = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        ref, frame = cap.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        fps = 0.0
        with torch.no_grad():
            while(True):
                t1 = time.time()
                ref, frame = cap.read()
                if not ref:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(np.uint8(frame))
                img = resize_image(frame, (512, 512), True)
                img = transforms.ToTensor()(img).unsqueeze(0)
                img = img.to(device)
                output = model(img)
                output = BBoxDecoder.decode_bbox(output[0], output[1], output[2], confidence=0.4)
                output = postprocess(output)[0].numpy()

                output[:, 0:4] = correct_boxes(output[:, 0:4], (512, 512), size)  # height, width
                frame = draw_box(frame, output[:, :4], output[:, -1], output[:, 4], class_names)
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                fps  = ( fps + (1./(time.time()-t1)) ) / 2
                print("fps= %.2f"%(fps))
                cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                if c == 27:
                    cap.release()
                    break
        cap.release()
        cv2.destroyAllWindows()