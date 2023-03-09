import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader
import cv2

import math
import random

from utils.utils import xywhn2xyxy
from utils.augmentations import random_perspective, augment_hsv, letterbox, DEFAULT_TRANSFORMS


class CenterNetDataset(Dataset):
    def __init__(self, data_root, img_size=512, isTrain=True, augment=False):
        self.augment = augment
        self.image_size = img_size
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        if isTrain:
            self.images_root = os.path.join(data_root, "train", "images")
        else:
            self.images_root = os.path.join(data_root, "val", "images")
        self.image_paths = [os.path.join(self.images_root, image_name) for image_name in os.listdir(self.images_root)] 
        self.image_paths.sort()
        self.annotations_paths = self._img2label_paths(self.image_paths)
        self.indices = range(len(self.image_paths))  # 每张图片的索引
        self.labels = self._load_labels()  # 一次性将所有 labels 加载进内存
        

    def _img2label_paths(self, image_paths):
        annotations_paths = []
        for image_path in image_paths:
            image_name = os.path.split(image_path)[-1]
            label_name = os.path.splitext(image_name)[0] + '.txt'
            annotations_paths.append(os.path.join(os.path.dirname(self.images_root), 'labels', label_name))
        return annotations_paths

    def _load_labels(self):
        labels = []
        for path in self.annotations_paths:
            labels.append(np.loadtxt(path, dtype=np.float32).reshape(-1, 5))
        return labels

    def __getitem__(self, index):
        """
        image(Tensor): shape[3, 512, 512]
        targets:(Tensor[N, 6]): [batch_index, class_label, x1, y1, x2, y2]
        """
        if self.augment:
            img, labels = self.load_mosaic(index)
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            shape = self.image_size
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        
        # Albumentations
        if self.augment:
            new = DEFAULT_TRANSFORMS(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            img, labels = new["image"], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

            # HSV color-space
            augment_hsv(img, hgain=0.1, sgain=0.7, vgain=0.4)

        # 第一个位置放当前图片在 batch 中的索引(在collate_fn中实现)
        nl = len(labels)
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out


    def __len__(self):
        return len(self.image_paths)

    def load_mosaic(self, index):
        """
        mosaic4 数据增强, 读取 4 张图片进行拼接, 并进行 random_perspective 数据增强
        """
        labels4 = []
        s = self.image_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + random.choices(self.indices, k=3)  # 再随机取 3 张图片
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, _, (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)

        labels4 = np.concatenate(labels4, 0)
        x = labels4[:, 1:]
        np.clip(x, 0, 2 * s, out=x)

        img4, labels4 = random_perspective(img4,
                                           labels4,
                                           degrees=0.0,
                                           translate=0.1,
                                           scale=0.5,
                                           shear=0.0,
                                           perspective=0.0,
                                           border=self.mosaic_border)

        return img4, labels4


    def load_image(self, i):
        """
        读取一张图片, 并将其长变 resize 到 self.image_size(保持长宽比不变)
        """
        im = cv2.imread(self.image_paths[i])
        assert im is not None, f"Image Not Found {self.image_paths[i]}"
        h0, w0 = im.shape[:2]
        r = self.image_size / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
        imgs = torch.stack(imgs)

        # add sample index to targets
        # to identify which image the boxes belong to
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        return imgs, targets


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt
    from draw_boxes_utils import draw_objs
    data = CenterNetDataset('./my_yolo_dataset', isTrain=True, augment=True)
    dataloader = DataLoader(data, 4, False, num_workers=4, collate_fn=data.collate_fn)

    for img, target in data:
        target = target.numpy()
        print(target)
        img = torchvision.transforms.ToPILImage()(img)
        img = draw_objs(img, target[:, 2:], np.ones_like(target[:, 1]))
        plt.imshow(img)
        plt.show()