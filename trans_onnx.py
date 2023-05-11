import torch
import onnx
import onnxruntime as rt
import cv2
import numpy as np

from networks.centernetplus import CenterNetPlus
from utils.augmentations import letterbox

image_path = './samples/imgs/111.jpeg'

def trans2onnx():
    model = CenterNetPlus(num_classes=1, backbone="r18")
    model.load_state_dict(torch.load("./run/DroneVsBirds_centernetplus_r18_best.pth"), strict=False)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    export_onnx_file = "DroneVsBirds_centernetplus_r18.onnx"
    torch.onnx.export(model, x, export_onnx_file, export_params=True, input_names=['input'], output_names=['output'])

def inference():
    model_path = './DroneVsBirds_centernetplus_r18.onnx'
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    session = rt.InferenceSession(model_path)
    img_origin = cv2.imread(image_path)
    img_origin = letterbox(img_origin, (512, 512), auto=False)[0]
    img_origin = img_origin.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img_origin).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    outputs = session.run(None, {'input': img})
    print(outputs)

if __name__ == '__main__':
    inference()