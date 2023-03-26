from utils.metrics import *
import torch
from torch.utils.data import DataLoader
from networks.centernetplus import CenterNetPlus
from networks.centernet import CenterNet
from utils.dataset import CenterNetDataset
from utils.boxes import postprocess, BBoxDecoder
from utils.utils import load_class_names
import tqdm

def evaluate(model, dataloader, device, label_path, plot=False):
    model.to(device)
    model.eval()
    class_names = load_class_names(label_path)

    stats = []
    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device).float() / 255

        targets = targets

        with torch.no_grad():
            outputs = model(imgs)
            outputs = BBoxDecoder.decode_bbox(outputs[0], outputs[1], outputs[2], 0.02)
            outputs = [o.cpu() for o in outputs]

            outputs = postprocess(outputs, nms_thres=0.4)

        stats += get_batch_statistics(outputs, targets)
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=class_names, plot=plot, save_dir='./run')
    print_eval_stats(ap, ap_class, class_names)
    return ap


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = CenterNetPlus(num_classes=20, backbone="r50")
    model.load_state_dict(torch.load("./run/20centernetplus_r50_best.pth"), strict=False)
    model.to(device)
    model.eval()

    dataset = CenterNetDataset('./my_yolo_dataset', isTrain=False, augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    evaluate(model, dataloader, device, label_path='./my_yolo_dataset/my_data_label.names', plot=True)
