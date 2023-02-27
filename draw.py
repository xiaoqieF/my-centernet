from networks.centernet import centernet_resnet18, centernet_darknet53
from networks.centernetplus import CenterNetPlus
from utils.dataset import CenterNetDataset
from utils.boxes import decode_bbox, postprocess
from torch.utils.data import DataLoader
import torch
from utils.draw_boxes_utils import draw_box
from utils.utils import load_class_names
import torchvision
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = CenterNetPlus(num_classes=2, backbone="r18")
    model.load_state_dict(torch.load("./run/centernetplus_r18_95.pth"))
    model.to(device)
    model.eval()

    data = CenterNetDataset("./DroneBirds", isTrain=False, augment=False)
    dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4, collate_fn=data.collate_fn)
    class_names = load_class_names("./DroneBirds/my_data_label.names")

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device).float() / 255
            targets = targets.to(device)

            outputs = model(imgs)
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], confidence=0.3)
            outputs = postprocess(outputs)[0]

            print(f"predictions: {outputs}")
            print(f"targets: {targets}")

            img = draw_box(torchvision.transforms.ToPILImage()(imgs.squeeze(0)), outputs[:, :4], outputs[:, -1], outputs[:, 4], class_names)
            plt.imshow(img)
            plt.show()