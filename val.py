from utils.metrics import *
import torch
from torch.utils.data import DataLoader
from networks.centernet import centernet_resnet50, centernet_darknet53
from utils.dataset import CenterNetDataset
from utils.transform import VAL_TRANSFORMS
from utils.hyp import HYP
from utils.boxes import decode_bbox, postprocess
from utils.utils import load_class_names
import tqdm

def evaluate(model, dataloader, device, plot=False):
    model.to(device)
    model.eval()
    class_names = load_class_names("./my_yolo_dataset/my_data_label.names")

    stats = []
    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device)
        targets = targets
        targets[:, 2:] *= HYP.input_size[0]

        with torch.no_grad():
            outputs = model(imgs)
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], 0.02)

            detections = postprocess(outputs, nms_thres=0.4)

        stats += get_batch_statistics(detections, targets)
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=class_names, plot=plot, save_dir='./run')
    print_eval_stats(ap, ap_class, class_names)
    return ap


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = centernet_darknet53()
    model.load_state_dict(torch.load("./run/centernet_darknet129.pth"))
    model.to(device)
    model.eval()

    dataset = CenterNetDataset('./my_yolo_dataset', isTrain=True, transform=VAL_TRANSFORMS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    evaluate(model, dataloader, device, plot=True)
