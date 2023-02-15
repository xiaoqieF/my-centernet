import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import numpy as np
from torch.utils.data import DataLoader
from .hyp import HYP


class CenterNetDataset(Dataset):
    def __init__(self, data_root, img_size=HYP.input_size, isTrain=True, transform=None):
        self.image_size = img_size
        if isTrain:
            self.images_root = os.path.join(data_root, "train", "images")
        else:
            self.images_root = os.path.join(data_root, "val", "images")
        self.image_paths = [os.path.join(self.images_root, image_name) for image_name in os.listdir(self.images_root)] 
        self.image_paths.sort()
        self.annotations_paths = []
        for image_path in self.image_paths:
            image_name = os.path.split(image_path)[-1]
            label_name = os.path.splitext(image_name)[0] + '.txt'
            self.annotations_paths.append(os.path.join(os.path.dirname(self.images_root), 'labels', label_name))
        self.transform = transform        

    def __getitem__(self, index):
        """
        image(Tensor): shape[3, 512, 512]
        targets:(Tensor[N, 6]): [batch_index, class_label, x1, y1, x2, y2]
        """
        image = np.array(Image.open(self.image_paths[index]), dtype=np.uint8)
        targets = np.loadtxt(self.annotations_paths[index], dtype=np.float32).reshape(-1, 5)
        if self.transform:
            image, targets = self.transform((image, targets))
        return image, targets
        

    def __len__(self):
        return len(self.image_paths)

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
    from transform import DEFAULT_TRANSFORMS, VAL_TRANSFORMS
    import torchvision
    from draw_boxes_utils import draw_objs
    import matplotlib.pyplot as plt
    data = CenterNetDataset('./my_yolo_dataset', isTrain=True, transform=VAL_TRANSFORMS)
    dataloader = DataLoader(data, 4, False, num_workers=4, collate_fn=data.collate_fn)

    for img, target in data:
        target[:, [2, 4]] *= img.shape[2]
        target[:, [3, 5]] *= img.shape[1]
        print(target)
        img = torchvision.transforms.ToPILImage()(img)
        plot_img = draw_objs(img, target[:, 2:], np.ones_like(target[:, 1]))
        plt.imshow(plot_img)
        plt.show()