import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

class AbsoluteLabels(object):
    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes

class RelativeLabels(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes

class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        img, boxes = data
        boxes = np.array(boxes)
        boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])
        
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],
            shape=img.shape
        )
        img, bounding_boxes = self.augmentations(
            image=img,
            bounding_boxes=bounding_boxes
        )
        bounding_boxes = bounding_boxes.clip_out_of_image()
        for i, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            boxes[i, 0] = box.label
            boxes[i, 1] = ((x1 + x2) / 2)
            boxes[i, 2] = ((y1 + y2) / 2)
            boxes[i, 3] = x2 - x1
            boxes[i, 4] = y2 - y1 

        return img, boxes

class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0, position="center-center"
            ).to_deterministic()
        ])

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        img = transforms.ToTensor()(img)
        bb_targets = torch.zeros((len(boxes), 6))
        # bb_targets[:, 0] 填 boxes 的样本序号
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)

        return img, bb_targets

class Resize(object):
    def __init__(self, out_shape=(512, 512)) -> None:
        self.out_shape = out_shape
        pass

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.out_shape, mode="nearest").squeeze(0)

        return img, boxes

class XYWH2XYXY(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data):
        img, boxes = data
        res_box = torch.zeros_like(boxes)
        res_box[:, 1] = boxes[:, 1]
        res_box[..., 2] = boxes[..., 2] - boxes[..., 4] / 2
        res_box[..., 3] = boxes[..., 3] - boxes[..., 5] / 2
        res_box[..., 4] = boxes[..., 2] + boxes[..., 4] / 2
        res_box[..., 5] = boxes[..., 3] + boxes[..., 5] / 2
        return img, res_box

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.01, 0.1]),
            iaa.Sharpen((0.0, 0.1)),
            # iaa.Affine(rotate=(-30, 30), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-60, 60)),
            iaa.Fliplr(0.5),
        ])

DEFAULT_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    DefaultAug(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize(),
    XYWH2XYXY()
])

VAL_TRANSFORMS = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    Resize(),
    XYWH2XYXY()
])