from utils.metrics import *
import torch
from torch.utils.data import DataLoader
from networks.centernet import centernet_resnet18, centernet_darknet53, centernet_resnet50
from networks.centernetplus import CenterNetPlus
from utils.dataset import CenterNetDataset
from utils.transform import VAL_TRANSFORMS
from utils.hyp import HYP
from utils.boxes import decode_bbox, postprocess
from utils.utils import load_class_names
import tqdm

def evaluate(model, dataloader, device, plot=False):
    model.to(device)
    model.eval()
    class_names = load_class_names("./DroneBirds/my_data_label.names")

    stats = []
    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        imgs = imgs.to(device).float() / 255
        targets = targets

        with torch.no_grad():
            outputs = model(imgs)
            outputs = decode_bbox(outputs[0], outputs[1], outputs[2], 0.02)
            outputs = [o.cpu() for o in outputs]

            outputs = postprocess(outputs, nms_thres=0.4)

        stats += get_batch_statistics(outputs, targets)
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=class_names, plot=plot, save_dir='./run')
    print_eval_stats(ap, ap_class, class_names)
    return ap


if __name__ == "__main__":
    device = torch.device("cuda:0")

    model = CenterNetPlus(num_classes=2)
    model.load_state_dict(torch.load("./run/centernetplus_r18.pth"))
    model.to(device)
    model.eval()

    dataset = CenterNetDataset('./DroneBirds', isTrain=False, augment=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    evaluate(model, dataloader, device, plot=True)
