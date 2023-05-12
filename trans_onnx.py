import torch
import onnx
import onnxruntime as rt
import cv2
import numpy as np
import os
import tensorrt as trt
from collections import OrderedDict
import matplotlib.pyplot as plt
import time

from networks.centernetplus import CenterNetPlus
from utils.augmentations import letterbox
from utils.boxes import postprocess, correct_boxes, BBoxDecoder
from utils.draw_boxes_utils import draw_box
from utils.utils import load_class_names

image_path = './samples/imgs/111.jpeg'

model_path='DroneVsBirds_centernetplus_r18.onnx'
engine_file_path = "DroneVsBirds_centernetplus_r18.trt"

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 32  # 4GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]

            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def trans2trt():
    return get_engine(model_path, engine_file_path)


def trans2onnx():
    model = CenterNetPlus(num_classes=1, backbone="r18")
    model.load_state_dict(torch.load("./run/DroneVsBirds_centernetplus_r18_best.pth"), strict=False)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    export_onnx_file = "DroneVsBirds_centernetplus_r18.onnx"
    torch.onnx.export(model, x, export_onnx_file, export_params=True, input_names=['input'], output_names=['output0', 'output1', 'output2'])

# 对 onnx 模型使用 onnxruntime 进行推理
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
    engine = trans2trt()
    context = engine.create_execution_context()
    bindings = OrderedDict()
    output_names = []   
    for i in range(engine.num_bindings):
        name = engine.get_tensor_name(i) # 获得输入输出的名字"images","output0"
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if engine.binding_is_input(i):  # 判断是否为输入
            if -1 in tuple(engine.get_tensor_shape(name)):  # dynamic get_binding_shape(0)->(1,3,640,640) get_binding_shape(1)->(1,25200,85)
                dynamic = True
                context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[2]))
            if dtype == np.float16:
                fp16 = True
        else:  # output
            output_names.append(name)  # 放入输出名字 output_names = ['output0']
        shape = tuple(context.get_tensor_shape(name))  # 记录输入输出shape
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).cuda()  # 创建一个全0的与输入或输出shape相同的tensor
        bindings[name] = [dtype, shape, im, int(im.data_ptr())]  # 放入之前创建的对象中   [dtype, shape, data, data_address]

    for i in range(5):
        img_origin = cv2.imread(image_path)
        img = letterbox(img_origin, (512, 512), auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        bindings['input'][2] = torch.from_numpy(img).cuda()
        bindings['input'][3] = bindings['input'][2].data_ptr()
        buffer = [d[3] for d in bindings.values()]   # cuda 变量地址，用于 tensorrt 推理
        t1 = time.time()
        context.execute_v2(buffer)
        print(f"time: {time.time() - t1}")
        output = [bindings['output0'][2], bindings['output1'][2], bindings['output2'][2]]
        output = BBoxDecoder.decode_bbox(output[0], output[1], output[2], confidence=0.3)
        output = postprocess(output)[0].numpy()

        output[:, 0:4] = correct_boxes(output[:, 0:4], (512, 512), (img_origin.shape[1], img_origin.shape[0]))  # height, width
        # print(f"predictions: {output}")

        class_names = load_class_names("./DroneVsBirds/my_data_label.names")
        img = draw_box(img_origin, output[:, :4], output[:, -1], output[:, 4], class_names)
        plt.imshow(img)
        plt.show()